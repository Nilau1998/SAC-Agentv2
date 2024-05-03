import torch

from agent.buffer import ReplayBuffer
from agent.base_agent import BaseAgent
from agent.networks.networks import ActorNetwork, CriticNetwork, ValueNetwork


class ContinousAgent(BaseAgent):
    def __init__(self, config, experiment_dir, input_dims, env):
        super().__init__(env)
        self.config = config
        self.gamma = config.agent.gamma
        self.tau = config.agent.tvn_parameter_modulation_tau
        self.memory = ReplayBuffer(
            config.agent.max_size, input_dims, self.get_n_actions()
        )
        self.batch_size = config.agent.batch_size

        self.actor = ActorNetwork(
            experiment_dir=experiment_dir,
            alpha=config.agent.learning_rate_alpha,
            input_dims=input_dims,
            max_action=self.get_max_actions(),
            n_actions=self.get_n_actions(),
            name="actor_network",
        )
        self.critic_1 = CriticNetwork(
            experiment_dir=experiment_dir,
            beta=config.agent.learning_rate_beta,
            input_dims=input_dims,
            n_actions=self.get_n_actions(),
            name="critic_network_1",
        )
        self.critic_2 = CriticNetwork(
            experiment_dir=experiment_dir,
            beta=config.agent.learning_rate_beta,
            input_dims=input_dims,
            n_actions=self.get_n_actions(),
            name="critic_network_2",
        )
        self.value = ValueNetwork(
            experiment_dir=experiment_dir,
            beta=config.agent.learning_rate_beta,
            input_dims=input_dims,
            name="value_network",
        )
        self.target_value = ValueNetwork(
            experiment_dir=experiment_dir,
            beta=config.agent.learning_rate_beta,
            input_dims=input_dims,
            name="target_value_network",
        )

        self.scale = config.agent.reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = (
                tau * value_state_dict[name].clone()
                + (1 - tau) * target_value_state_dict[name].clone()
            )

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * torch.functional.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * torch.functional.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * torch.functional.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
