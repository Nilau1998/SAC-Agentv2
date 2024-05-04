import torch
from agent.networks.base_network import BaseNetwork


class ActorNetwork(BaseNetwork):
    def __init__(
        self,
        experiment_dir,
        alpha,
        input_dims,
        max_action,
        fc1_dims=256,
        fc2_dims=256,
        n_actions=2,
        name="actor_network",
    ):
        super().__init__(name, experiment_dir)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.reparam_noise = 1e-6

        # Define layers
        self.fc1 = torch.nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = torch.nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mean = torch.nn.Linear(self.fc2_dims, self.n_actions)
        self.std = torch.nn.Linear(self.fc2_dims, self.n_actions)

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        prob = torch.nn.functional.relu(self.fc1(state))
        prob = torch.nn.functional.relu(self.fc2(prob))

        mean = self.mean(prob)
        std = self.std(prob)

        return mean, std

    def sample_normal(self, state, reparameterize=True):
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5

        mean, std = self.forward(state)

        log_std = torch.tanh(std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)

        if reparameterize:
            actions = normal.rsample()
        else:
            actions = normal.sample()

        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        log_probs = normal.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(0, keepdim=True)

        return action, log_probs


class CriticNetwork(BaseNetwork):
    def __init__(
        self,
        experiment_dir,
        beta,
        input_dims,
        n_actions,
        fc1_dims=256,
        fc2_dims=256,
        name="critic_network",
    ):
        super(CriticNetwork, self).__init__(name, experiment_dir)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # Define layers
        self.fc1 = torch.nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = torch.nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = torch.nn.Linear(self.fc2_dims, 1)

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = torch.nn.functional.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = torch.nn.functional.relu(action_value)

        q = self.q(action_value)

        return q


class ValueNetwork(BaseNetwork):
    def __init__(
        self,
        experiment_dir,
        beta,
        input_dims,
        fc1_dims=256,
        fc2_dims=256,
        name="value_network",
    ):
        super().__init__(name, experiment_dir)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        # Define layers
        self.fc1 = torch.nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = torch.nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = torch.nn.Linear(self.fc2_dims, 1)

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = torch.nn.functional.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = torch.nn.functional.relu(state_value)

        v = self.v(state_value)

        return v
