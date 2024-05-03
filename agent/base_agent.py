from gymnasium.spaces import Box

class BaseAgent:
    def __init__(self, env):
        self.env = env

    def get_n_actions(self):
        # Check if Box or Discrete, then return the action space size (n_actions)
        if isinstance(self.env.action_space, Box):
            return self.env.action_space.shape[0]
        else:
            return self.env.action_space.n

    def get_max_actions(self):
        # Check if Box or Discrete, then return the max_action value for the actor network
        if isinstance(self.env.action_space, Box):
            return self.env.action_space.high
        else:
            raise NotImplementedError