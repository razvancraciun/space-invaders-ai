from nn import NN
import gym
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from buffer import ReplayBuffer

class Agent:
    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')
        self.epsilon = 0.9
        self.discount = 0.95
        self.nn = NN(self.env.action_space)
        self.frame_stack = []
        self.stack_size = 3
        self.buffer = ReplayBuffer(100)
        self.render_interval = 10


    def train(self, episodes):
        for episode in range(episodes):
            self.train_episode(episode)
            self.train_buffer()


    def train_buffer(self):
        items = np.array(self.buffer.items)
        states_ = items[:,3][:]
        states_.reshape(len(states_), 210, 160, 1)
        
        # state1, action, reward, new_state, Q = items[:,0][:], items[:,1][:], items[:,2][:], items[:,:,3], items[:,4][:]
        y = self.nn.model.predict(states_)
        # target = reward + self.discount * np.max(y)
        # Q[action] = target
        # self.nn.model.fit(state1, Q)

    def train_episode(self, episode):
        print(f'Playing episode {episode}')
        done = False
        self.init_stack(self.preprocess(self.env.reset()))
        state = self.stack_frame()
        while not done:
            Q = self.nn.model.predict(state.reshape(1,*state.shape))[0]
            action = np.argmax(Q)
            new_state, reward, done, _ = self.env.step(action)
            self.add_frame(self.preprocess(new_state))
            new_state = self.stack_frame()
            if episode % self.render_interval == 0 and episode != 0:
                self.env.render()
            self.buffer.add( [state, action, reward, new_state, Q] )
            state = new_state


    def init_stack(self, frame):
        self.frame_stack = [frame for _ in range(self.stack_size)]

    def add_frame(self, frame):
        self.frame_stack.pop(0)
        self.frame_stack.append(frame)

    def stack_frame(self):
        coef = 0.55
        result = self.frame_stack[-1] * coef
        for frame in reversed(self.frame_stack[:-1]):
            coef -= 0.22
            result += frame * coef
        # plt.imshow(result.reshape(*result.shape[1:-1]), cmap='gray')
        # plt.show()
        return result

    def preprocess(self, state):
        def to_grayscale(img):
            r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
            return 0.2989 * r + 0.587 * g + 0.114 * b
        result = to_grayscale(state)
        result = result.reshape(*result.shape,1)
        return result
