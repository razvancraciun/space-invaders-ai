from nn import init_model
from buffer import ReplayBuffer
import gym
import numpy as np
import json

class Agent:
    def __init__(self, config_file, input_shape, n_actions):
        config = json.loads(open(config_file).read())
        self.epsilon = config['epsilon']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_dec = config['epsilon_dec']
        self.gamma = config['gamma']
        self.input_shape = input_shape
        self.batch_size = config['batch_size']

        self.action_space = [i for i in range(n_actions)]        

        self.model = init_model(input_shape, n_actions, config['learning_rate'], config['loss'])
        self.buffer = ReplayBuffer(config['memory_size'], input_shape, n_actions)

        self.save_path = config['save_path']
    

    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.model.predict(state)
            action = np.argmax(actions)
        return action


    def train(self):
        if self.buffer.count < self.batch_size:
            return
        from_states, actions, rewards, to_states, terminals = self.buffer.sample(self.batch_size)

        q = self.model.predict(from_states)
        q_next = self.model.predict(to_states)

        q_target = q.copy()
        batch_index = np.arrange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)*terminals

        self.model.fit(from_states, q_target, verbose=False)

        self.epsilon = self.epsilon * self.epsilon_dec \
            if self.epsilon > self.epsilon_min else self.epsilon_min
        

    def save(self, index):
        self.model.save(self.save_path + f'ckpt{index}.h5')

    def load(self, index):
        self.model.load(self.save_path + f'ckpt{index}.h5')

    