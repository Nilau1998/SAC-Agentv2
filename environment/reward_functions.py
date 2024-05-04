import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os


class RewardFunction:
    """
    Class that implements a bunch of reward functions that can be used to calculate the rewards for the agent. The reward functions need a track_width which describes the area the boat should stay in on the y axis. Additionally the reward functions have a x_pos_multiplier that gives extra reward the further the boat gets this parameter however can be turned of by just not setting it and leaving it in its default being 0. The linear, quadratic.. only matters for the y axis. Linear would mean that the middle of the track gives max points and it linearely decreases towards the edge to 0. Additional points are given the futher the agent drives the boat towards the goal line.
    """

    def __init__(self, config, experiment_dir, x_a=1, x_b=1, y_a=1, y_b=1):
        self.track_width = config.boat_env.track_width
        self.track_length = config.boat_env.goal_line
        self.experiment_dir = experiment_dir
        self.x_a = x_a
        self.x_b = x_b
        self.y_a = y_a
        self.y_b = y_b

    def visualize_reward_field(self, f_x, f_y):
        reward_field_file = os.path.join(self.experiment_dir, "reward_field.png")
        if not os.path.exists(reward_field_file):
            n = 256
            x = np.linspace(0.0, self.track_length, n)
            y = np.linspace(-self.track_width, self.track_width, n)
            X, Y = np.meshgrid(x, y)
            sns.set_style("whitegrid", {"axes.grid": True, "axes.edgecolor": "black"})
            Z = f_x(X) - f_y(Y)
            plt.pcolormesh(X, Y, Z, cmap=cm.Blues)
            plt.colorbar()
            plt.xlabel("Position x")
            plt.ylabel("Position y")
            plt.title("Rewardverteilung")
            plt.savefig(reward_field_file, dpi=400)
            plt.close("all")

    def exponential_reward(self, position):
        # Used desmos graphing to get the track_width / 6 parameter.
        # https://www.desmos.com/calculator/f453o0sxmf
        x = position[0]
        y = position[1]

        def f_x(x):
            return 0

        def f_y(y):
            return (np.abs(y) / self.track_width) / (
                1
                + np.exp(
                    (-self.y_a / self.y_b) * (np.abs(y) - (self.track_width * 0.2))
                )
            )

        self.visualize_reward_field(f_x, f_y)

        return f_x(x) - f_y(y)
