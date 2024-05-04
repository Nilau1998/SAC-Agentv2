import torch


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_memory = torch.zeros(
            (self.mem_size, *input_shape), dtype=torch.float32
        ).to(self.device)
        self.new_state_memory = torch.zeros(
            (self.mem_size, *input_shape), dtype=torch.float32
        ).to(self.device)
        self.action_memory = torch.zeros(
            (self.mem_size, n_actions), dtype=torch.float32
        ).to(self.device)
        self.reward_memory = torch.zeros(self.mem_size, dtype=torch.float32).to(
            self.device
        )
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool).to(
            self.device
        )

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = torch.randint(0, max_mem, (batch_size,))

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
