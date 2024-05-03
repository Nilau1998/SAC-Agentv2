import os
import torch

class BaseNetwork(torch.nn.Module):
    def __init__(self, name, experiment_dir):
        super(BaseNetwork, self).__init__()
        self.name = name
        self.experiment_dir = experiment_dir
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def return_summed_weights(self):
        total_sum = 0
        for layer in self.children():
            if isinstance(layer, torch.nn.Linear):
                total_sum += torch.sum(layer.state_dict()["weight"])
        print(f"{self.name}: {total_sum:.2f}")