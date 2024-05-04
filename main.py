import os
import numpy as np
import csv
import pygame

from utils.config_reader import get_config, get_experiment_config
from utils.build_experiment import Experiment
from environment.boat_env import BoatEnv
from postprocessing.recorder import Recorder
from agent.continuous_agent import ContinuousAgent
from rendering.renderer import RenderBoatEnv


class ControlCenter:
    def __init__(self, subdir=None):
        self.config = get_config("original_config.yaml")
        self.subdir = subdir
        self.experiment_overview_file = os.path.join(
            "experiments", self.subdir, "overview.csv"
        )
        self.terminations_file = None
        self.experiment = None

    def train_model(self, tuner=False):
        self.experiment = Experiment(subdir=self.subdir)
        self.experiment.save_configs()
        self.terminations_file = os.path.join(
            self.experiment.experiment_dir, "terminations.csv"
        )

        if tuner:
            self.config = get_experiment_config(
                self.experiment.experiment_dir, "tuned_configs.yaml"
            )

        env = BoatEnv(self.config, self.experiment)

        recorder = Recorder(env)

        if int(self.config.pygame.render) == 1:
            renderer = RenderBoatEnv(self.config)

        agent = ContinuousAgent(
            config=self.config,
            experiment_dir=self.experiment.experiment_dir,
            input_dims=env.observation_space.shape,
            env=env,
        )

        best_score = float("-inf")
        score_history = []
        load_checkpoint = False

        if load_checkpoint:
            agent.load_models()

        for i in range(self.config.base_settings.n_games):
            observation = env.reset()
            done = False
            score = 0

            recorder.create_csvs(i)
            while not done:
                recorder.write_data_to_csv()
                action = agent.choose_action(observation)
                observation_, reward, done, self.info = env.step(action)
                score += reward
                if self.info["termination"] == "reached_goal":
                    agent.remember(observation, action, reward, observation_, True)
                else:
                    agent.remember(observation, action, reward, observation_, False)
                if not load_checkpoint:
                    agent.learn()
                observation = observation_

                # Render the environment
                if int(self.config.pygame.render) == 1:
                    renderer.set_state(env.return_all_data())
                    renderer.set_wind(env.boat.get_wind())
                    renderer.render()
            print(self.info["termination"])
            recorder.write_info_to_csv()
            recorder.write_winds_to_csv()
            score_history.append(score)
            avg_score = np.mean(
                score_history[-self.config.base_settings.avg_lookback :]
            )

            if score > best_score:
                best_score = score

            if score > avg_score:
                if not load_checkpoint:
                    agent.save_models()

            if int(self.config.pygame.render) == 1:
                # Kill pygame window
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        renderer.close()

        if not os.path.exists(self.experiment_overview_file):
            with open(self.experiment_overview_file, "x") as csv_file:
                writer = csv.writer(csv_file, delimiter=";")
                writer.writerow([self.experiment.experiment_name, best_score])
        else:
            with open(self.experiment_overview_file, "a") as csv_file:
                writer = csv.writer(csv_file, delimiter=";")
                writer.writerow([self.experiment.experiment_name, best_score])

        with open(self.terminations_file, "x") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(self.info.keys())
            writer.writerow(self.info.values())


if __name__ == "__main__":
    control_center = ControlCenter(subdir="test")
    control_center.train_model()
