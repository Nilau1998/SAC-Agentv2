from dataclasses import dataclass
from pathlib import Path
import yaml
from dotmap import DotMap
import random


class HPTuner:
    def __init__(self):
        self.base_config = yaml.safe_load(
            Path('configs', 'hp_configs.yaml').read_text())
        self.hp_configs_dotmap = DotMap(self.base_config)
        self.hpset = HPSet(
            alpha=self.alpha(),
            beta=self.beta(),
            gamma=self.gamma(),
            tau=self.tau()
        )

    def alpha(self):
        return round(random.uniform(
            self.hp_configs_dotmap.agent.alpha_min,
            self.hp_configs_dotmap.agent.alpha_max
        ), 4)

    def beta(self):
        return round(random.uniform(
            self.hp_configs_dotmap.agent.beta_min,
            self.hp_configs_dotmap.agent.beta_max
        ), 4)

    def gamma(self):
        self.hp_configs_dotmap.agent.test = 5
        return round(random.uniform(
            self.hp_configs_dotmap.agent.gamma_min,
            self.hp_configs_dotmap.agent.gamma_max
        ), 4)

    def tau(self):
        return round(random.uniform(
            self.hp_configs_dotmap.agent.tau_min,
            self.hp_configs_dotmap.agent.tau_max
        ), 4)

    def generate_tuned_config_file(self, original_config, experiment_dir):
        original_config.agent.learning_rate_alpha = self.hpset.alpha
        original_config.agent.learning_rate_beta = self.hpset.beta
        original_config.agent.gamma = self.hpset.gamma
        original_config.agent.tvn_parameter_modulation_tau = self.hpset.tau
        config_dict = original_config.toDict()
        with open(Path(experiment_dir, 'configs', 'tuned_configs.yaml'), 'w') as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=False)


@dataclass
class HPSet:
    alpha: float
    beta: float
    gamma: float
    tau: float


if __name__ == '__main__':
    hptuner = HPTuner('hp_configs.yaml')
