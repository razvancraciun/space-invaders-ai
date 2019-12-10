from nn import NN
import gym
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

class Agent:
    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')
        self.epsilon = 0.9
        self.discount = 0.95
        self.nn = NN(self.env.action_space)
        self.frame_stack = []
        self.stack_size = 4


    def train(self, episodes):
        for episode in range(episodes):
            self.train_episode(episode)

    def train_episode(self, episode):
        done = False
        state = self.preprocess(self.env.reset())
        self.init_stack(state)
        while not done:
            Q = self.nn.model.predict(self.stack_frame())
            action = np.argmax(Q)
            new_state, reward, done, _ = self.env.step(action)
            new_state = self.add_frame(self.preprocess(new_state))
            self.env.render()
            target = reward + self.discount * max(self.nn.model.predict(self.stack_frame()))

            state = new_state


    def init_stack(self, frame):
        self.frame_stack = [frame for _ in range(self.stack_size)]

    def add_frame(self, frame):
        self.frame_stack.pop(0)
        self.frame_stack.append(frame)

    def stack_frame(self):
        coef = 0.4
        result = self.frame_stack[-1] * coef
        for frame in reversed(self.frame_stack[:-1]):
            coef -= 0.1
            result += frame * coef
        # plt.imshow(result.reshape(*result.shape[1:-1]), cmap='gray')
        # plt.show()
        return result

    def preprocess(self, state):
        def to_grayscale(img):
            r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
            return 0.2989 * r + 0.587 * g + 0.114 * b
        result = to_grayscale(state)
        result = result.reshape(1, *result.shape,1)
        return result
