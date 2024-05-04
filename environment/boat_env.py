import gymnasium
import numpy as np
import torch

from environment.reward_functions import RewardFunction
from environment.wind import Wind
from environment.control_theory.blocks import Integrator


class BoatEnv(gymnasium.Env):
    def __init__(self, config, experiment=None):
        self.config = config
        self.experiment_dir = experiment.experiment_dir

        self.action = [0]
        self.reward = 0
        self.boat = Boat(self.config)
        self.reward_function = RewardFunction(
            config=config,
            experiment_dir=self.experiment_dir,
            x_a=1,
            x_b=0.4,
            y_a=0.03,
            y_b=3.4,
        )

        self.info = {
            "termination": "",
            "reached_goal": 0,
            "out_of_bounds": 0,
            "out_of_fuel": 0,
            "rudder_broken": 0,
            "timeout": 0,
            "episode_reward": 0,
        }

        # Define action space, following actions can be choosen:
        # rudder [rad]
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, dtype=np.float32)

        # Define obeservation space
        # Following states are observed:
        # s_x, v_x, a_x
        # s_y, v_y, a_y
        # s_r, v_r, a_r
        # rudder_angle, fuel
        self.low_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.high_state = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

    def step(self, action):
        self.action = action
        self.boat.t += self.boat.dt
        self.boat.fuel -= 1

        if self.config.base_settings.test_mode == 0:
            self.boat.rudder_angle += action[0].cpu().numpy() / 10

        self.boat.run_model_step()

        self.state = self.boat.return_state()

        # Reward calculation
        self.reward = self.reward_function.exponential_reward(
            position=[self.boat.s_x, self.boat.s_y],
        )

        # Termination logic
        done = False
        if self.boat.s_x >= self.config.boat_env.goal_line:
            done = True
            self.info["termination"] = "reached_goal"
            self.info["reached_goal"] += 1
            self.reward += 1000
        elif abs(self.boat.s_y) > self.boat.out_of_bounds or self.boat.s_x < 0:
            done = True
            self.info["termination"] = "out_of_bounds"
            self.info["out_of_bounds"] += 1
        elif self.boat.fuel < 0:
            done = True
            self.info["termination"] = "out_of_fuel"
            self.info["out_of_fuel"] += 1
        elif self.boat.t_max <= self.boat.t:
            done = True
            self.info["termination"] = "timeout"
            self.info["timeout"] += 1
        elif self.boat.rudder_angle > np.pi / 3 or self.boat.rudder_angle < -np.pi / 3:
            done = True
            self.info["termination"] = "rudder_broken"
            self.info["rudder_broken"] += 1

        if self.boat.rudder_angle > np.pi / 4 or self.boat.rudder_angle < -np.pi / 4:
            self.reward -= np.abs(self.boat.rudder_angle) * 100

        if np.abs(self.boat.s_r) > np.pi / 2:
            self.reward -= 1

        self.info["episode_reward"] += self.reward

        return self.state, self.reward, done, self.info

    def reset(self):
        self.boat = Boat(self.config)
        self.info["episode_reward"] = 0

        self.state = self.boat.return_state()

        return self.state

    def render(self):
        pass

    def return_all_data(self):
        data = {
            "boat_position_x": self.boat.s_x,
            "boat_position_y": self.boat.s_y,
            "boat_velocity_x": self.boat.v_x,
            "boat_velocity_y": self.boat.v_y,
            "boat_angle": self.boat.s_r,
            "action_rudder": self.action[0],
            "reward": self.reward,
            "rudder_angle": self.boat.rudder_angle,
            "n": self.boat.n,
        }
        return data


class Boat:
    def __init__(self, config):
        self.config = config

        # Construct tensor once, then update values
        self.state_return = torch.zeros(11, dtype=torch.float32)

        self.s_y_start = np.random.randint(
            -int(self.config.boat_env.track_width * 0.8),
            int(self.config.boat_env.track_width * 0.8),
        )

        self.t = 0
        self.dt = config.base_settings.dt
        self.t_max = config.base_settings.t_max
        self.index = 0  # Used to access arrays since dt is a float not an int
        self.wind = Wind(config)

        self.a_x_integrator = Integrator(initial_value=3)
        self.a_x_integrator.dt = self.dt
        self.v_x_integrator = Integrator()
        self.v_x_integrator.dt = self.dt

        self.a_y_integrator = Integrator()
        self.a_y_integrator.dt = self.dt

        if int(self.config.base_settings.experiment) == 2:
            self.v_y_integrator = Integrator(initial_value=self.s_y_start)
        else:
            self.v_y_integrator = Integrator()
        self.v_y_integrator.dt = self.dt

        self.a_r_integrator = Integrator()
        self.a_r_integrator.dt = self.dt
        self.v_r_integrator = Integrator()
        self.v_r_integrator.dt = self.dt

        # Boat
        self.n = 20
        self.rudder_angle = 0
        self.fuel = config.boat.fuel

        self.a_x = 0
        self.v_x = 0
        self.s_x = 0

        self.a_y = 0
        self.v_y = 0
        self.s_y = 0

        self.a_r = 0
        self.v_r = 0
        self.s_r = 0

        self.v = 0
        self.drift_angle = 0
        self.turning_rate = 0

        self.get_kinematics()

        self.out_of_bounds = (
            self.config.boat_env.track_width
            + self.config.boat_env.boat_out_of_bounds_offset
        )

    def run_model_step(self):
        self.eom_longitudinal()
        self.v_x = self.a_x_integrator.integrate_signal(self.a_x)
        self.eom_transverse()
        self.v_y = self.a_y_integrator.integrate_signal(self.a_y)
        self.eom_yawning()
        self.v_r = self.a_r_integrator.integrate_signal(self.a_r)
        self.get_kinematics()
        self.index += 1

    def eom_longitudinal(self):
        params = self.config.boat

        # Resistance F_R
        F_R = (
            np.square(self.v_x)
            * params.c_r_front
            * 0.5
            * params.rho
            * params.boat_area_front
        )

        # Thrust F_T
        v_x_w = self.v_x * (1 - params.wake_friction)
        J = 0
        if self.n != 0:
            J = v_x_w / (self.n * params.propeller_diameter)
        KT = np.sin(J)
        F_T = (
            KT
            * np.square(self.n)
            * params.rho
            * np.power(params.propeller_diameter, 4)
            * (1 - params.thrust_deduction)
        )

        # Centrifugal force F_C
        F_C = self.v_y * (params.boat_m + params.boat_m_y) * self.v_r

        # Wind
        wind_v_sign = np.sign(self.wind.get_wind(self.index)[0])
        F_W_unangled = (
            np.square(self.wind.get_wind(self.index)[0])
            * wind_v_sign
            * params.c_r_front
            * 0.5
            * params.rho
            * params.boat_area_front
        )

        F_W = F_W_unangled * np.cos(self.wind.get_wind(self.index)[1])

        self.a_x = (-F_R + F_T + F_C + F_W) / (params.boat_m + params.boat_m_x)

    def eom_transverse(self):
        params = self.config.boat
        v_y_sign = np.sign(self.v_y)

        # Resistance F_R
        F_R = (
            np.square(self.v_y)
            * params.c_r_side
            * 0.5
            * params.rho
            * params.boat_area_side
            * v_y_sign
        )

        # Rudder force F_RU
        F_RU = (
            np.square(self.v_x)
            * params.c_r_front
            * 0.5
            * params.rho
            * params.rudder_area
        )
        F_RU = np.sin(self.rudder_angle) * F_RU

        # Centrifugal force F_C
        F_C = self.v_x * (params.boat_m + params.boat_m_x) * self.v_r

        # Wind
        wind_v_sign = np.sign(self.wind.get_wind(self.index)[0])
        F_W_unangled = (
            np.square(self.wind.get_wind(self.index)[0])
            * wind_v_sign
            * params.c_r_side
            * 0.5
            * params.rho
            * params.boat_area_side
        )

        F_W = F_W_unangled * np.sin(self.wind.get_wind(self.index)[1])

        self.a_y = (-F_R + F_RU + F_C + F_W) / (params.boat_m + params.boat_m_y)

    def eom_yawning(self):
        params = self.config.boat
        v_r_sign = np.sign(self.v_r)
        v_x_sign = np.sign(self.v_x)

        # Momentum from hull M_hull
        M_hull = (
            np.square(self.v_r)
            * params.c_r_side
            * 0.5
            * params.rho
            * params.boat_area_side
            * params.boat_l
            * 5
            * v_r_sign
        )

        # Moment from rudder M_rudder
        M_rudder = (
            np.square(self.v_x)
            * params.c_r_side
            * 0.5
            * params.rho
            * params.rudder_area
            * np.sin(self.rudder_angle)
            * (params.boat_b / 2)
            * v_x_sign
        )

        self.a_r = (-M_hull + M_rudder) / (params.boat_I + params.boat_Iz)

    def get_kinematics(self):
        params = self.config.boat
        # v
        self.v = np.sqrt(np.square(self.v_x) + np.square(self.v_y))

        # drift_angle
        self.drift_angle = np.arctan2(self.v_x, self.v_y)

        # turning rate
        self.turning_rate = 0
        if self.v != 0:
            self.turning_rate = (self.v_r * params.boat_l) / self.v

        # heading
        self.s_r = self.v_r_integrator.integrate_signal(self.v_r)

        # s_x in new coordinate system
        direction = self.drift_angle - self.s_r
        v_x_new = np.sin(direction) * self.v
        self.s_x = self.v_x_integrator.integrate_signal(v_x_new)

        # s_y in new coordinate system
        v_y_new = np.cos(direction) * self.v
        self.s_y = self.v_y_integrator.integrate_signal(v_y_new)

    def return_state(self):
        self.state_return[0] = self.normalize(
            self.s_x, 0, self.config.boat_env.goal_line
        )
        self.state_return[1] = self.normalize(self.v_x, 0, 5)
        self.state_return[2] = self.normalize(self.a_x, 0, 0.025)
        self.state_return[3] = self.normalize(
            self.s_y,
            -self.config.boat_env.track_width,
            self.config.boat_env.track_width,
        )
        self.state_return[4] = self.normalize(self.v_y, 0, 2)
        self.state_return[5] = self.normalize(self.a_y, 0, 0.37)
        self.state_return[6] = self.normalize(self.s_r, 0, 2 * np.pi)
        self.state_return[7] = self.normalize(self.v_r, 0, 8.5e-3)
        self.state_return[8] = self.normalize(self.rudder_angle, -np.pi / 3, np.pi / 3)
        self.state_return[9] = self.normalize(self.fuel, 0, self.config.boat.fuel)

        return self.state_return

    def get_wind(self):
        return self.wind.get_wind(self.index)

    def normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)
