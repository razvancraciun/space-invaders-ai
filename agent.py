from nntorch import NN
import gym
import numpy as np
from PIL import Image


class Agent:
    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')
        self.epsilon = 0.9
        self.discount = 0.95
        self.nn = NN(self.env.action_space)
        print(self.nn.parameters())


    def train(self):
        done = False
        state = self.preprocess(self.env.reset())
        while not done:
            Q = self.nn.forward(state).data.numpy()
            action = np.argmax(Q)
            new_state, reward, done, _ = self.env.step(action)
            new_state = self.preprocess(new_state)
            self.env.render()

            target = reward + self.discount * max(self.nn.forward(new_state))

            self.nn.backward(target)

            state = new_state



    def preprocess(self, state):
        def to_grayscale(img):
            r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
            return 0.2989 * r + 0.587 * g + 0.114 * b
        result = to_grayscale(state)
        result = result.reshape(1,1, *result.shape)
        return result
