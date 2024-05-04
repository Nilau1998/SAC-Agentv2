import csv
import os
import numpy as np


class Recorder:
    """
    Records the data of an episode and stores it into a csv for later use. Can for example be used by the env renderer.
    """

    def __init__(self, env):
        self.env = env
        self.experiment_dir = env.experiment_dir
        self.data_file = None
        self.info_file = None

    def create_csvs(self, episode_index):
        self.data_file = os.path.join(
            self.experiment_dir, "episodes", f"episode_{episode_index}_data.csv"
        )
        if not os.path.exists(self.data_file):
            with open(self.data_file, "x") as csv_file:
                writer = csv.writer(csv_file, delimiter=";")
                writer.writerow(self.env.return_all_data().keys())

        self.info_file = os.path.join(self.experiment_dir, "episodes", "info.csv")
        if not os.path.exists(self.info_file):
            with open(self.info_file, "x") as csv_file:
                writer = csv.writer(csv_file, delimiter=";")
                writer.writerow(self.env.info.keys())

    def write_data_to_csv(self):
        with open(self.data_file, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(self.env.return_all_data().values())

    def write_info_to_csv(self, extra_info=None):
        if extra_info != None:
            self.env.info["extra_info"] = extra_info
        with open(self.info_file, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(self.env.info.values())

    def write_winds_to_csv(self):
        wind = np.column_stack(
            (self.env.boat.wind.wind_velocity, self.env.boat.wind.wind_angle)
        )
        self.wind_file = os.path.join(self.experiment_dir, "episodes", "wind.csv")
        if not os.path.exists(self.wind_file):
            with open(self.wind_file, "x") as csv_file:
                writer = csv.writer(csv_file, delimiter=";")
                writer.writerow(["wind_velocity", "wind_angle"])
                for _ in wind:
                    writer.writerow(_)
