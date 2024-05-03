import os
import time
from distutils.dir_util import copy_tree

from utils.hyperparameter_tuner import HPTuner
from utils.config_reader import get_experiment_config


class Experiment:
    def __init__(self, experiment_name="experiment", subdir=None) -> None:
        self.timestamp = time.strftime("_%m-%d_%H-%M-%S%f")[:-2]
        self.experiments_dir = "experiments"
        if subdir != None:
            self.experiments_dir = os.path.join("experiments", subdir)
        self.tuner = HPTuner()

        # Create parent directiory for all experiments
        if not os.path.exists(self.experiments_dir):
            os.makedirs(self.experiments_dir)

        self.experiment_name = experiment_name + self.timestamp
        self.experiment_dir = os.path.join(self.experiments_dir, self.experiment_name)

        # Create subfolders in current experiment
        os.makedirs(os.path.join(self.experiment_dir, "plots"))
        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"))
        os.makedirs(os.path.join(self.experiment_dir, "configs"))
        os.makedirs(os.path.join(self.experiment_dir, "rendering"))
        os.makedirs(os.path.join(self.experiment_dir, "episodes"))

        print(f"Created: {self.experiment_dir}")

    def save_configs(self):
        experiment_configs = os.path.join(self.experiment_dir, "configs")
        copy_tree("configs", experiment_configs)
        self.tuner.generate_tuned_config_file(
            get_experiment_config(self.experiment_dir, "original_config.yaml"),
            self.experiment_dir,
        )
