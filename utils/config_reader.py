from pathlib import Path
import yaml
from dotmap import DotMap


def get_config(config_file):
    raw_config = yaml.safe_load(Path("configs", config_file).read_text())
    return DotMap(raw_config)


def get_experiment_config(experiment_dir, config_file):
    raw_config = yaml.safe_load(
        Path(experiment_dir, "configs", config_file).read_text())
    return DotMap(raw_config)
