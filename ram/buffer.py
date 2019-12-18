import numpy as np

class ReplayBuffer:
    def __init__(self, size, state_shape, n_actions):
        self.size = size
        self.count = 0
        self.from_states = np.zeros( (self.size, state_shape) ) 
        self.to_states = np.zeros( (self.size, state_shape) )
        self.actions = np.zeros(self.size, dtype=np.int8)
        self.rewards = np.zeros(self.size)
        self.terminals = np.zeros(self.size)

    def store(self, from_state, action, reward, to_state, done):
        index = self.count % self.size
        self.from_states[index] = from_state
        self.to_states[index] = to_state
        self.actions[index] = action
        self.rewards[index] = self.size
        self.terminals[index] = 1 - int(done)
        self.count += 1

    def sample(self, batch_size):
        size = min(self.count, self.size)
        batch = np.random.choice(size, batch_size)

        from_states = self.from_states[batch]
        to_states = self.to_states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        terminals = self.terminals[batch]

        return from_states, actions, rewards, to_states, terminals 


