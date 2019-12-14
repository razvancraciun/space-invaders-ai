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
        self.epsilon = 0.99
        self.discount = 0.95
        self.nn = NN(self.env.action_space)
        self.frame_stack = []
        self.stack_size = 3
        self.buffer = ReplayBuffer(4000)
        self.render_interval = 3


    def train(self, episodes):
        for episode in range(episodes):
            self.train_episode(episode)
            self.train_buffer()


    def train_buffer(self):
        items = np.array(self.buffer.items)
        # unused actions?
        states, actions, rewards, states_ = np.array(list(items[:,0])), np.array(list(items[:,1])), \
            np.array(list(items[:,2])), np.array(list(items[:,3]))
        Qpred = self.nn.model.predict(states)
        Qnext = self.nn.model.predict(states_)
        max_actions = np.argmax(Qnext, 1)
        Qtarget = Qpred
        Qtarget[:, max_actions] = rewards + self.discount * np.max(Qnext, 1)
        self.nn.model.fit(states, Qtarget, batch_size=100, workers=5)

    def train_episode(self, episode):
        print(f'Playing episode {episode}')
        done = False
        self.init_stack(self.preprocess(self.env.reset()))
        state = self.stack_frame()
        while not done:
            action = np.random.randint(0, self.env.action_space.n)
            if np.random.rand() > self.epsilon:
                Q = self.nn.model.predict(state.reshape(1,*state.shape))[0]
                action = np.argmax(Q)
            new_state, reward, done, _ = self.env.step(action)
            self.add_frame(self.preprocess(new_state))
            new_state = self.stack_frame()
            # if episode % self.render_interval == 0 and episode != 0:
            #     self.env.render()
            self.buffer.add( [state, action, reward, new_state] )
            state = new_state
        self.epsilon *= 0.95
        if self.epsilon < 0.01:
          self.epsilon = 0.01


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

    def play_episode(self):
      done = False
      self.init_stack(self.preprocess(self.env.reset))
      state = self.stack_frame()
      done = False
      while not done:
          Q = self.nn.model.predict(state.reshape(1,*state.shape))[0]
          action = np.argmax(Q)
          new_state, reward, done, _ = self.env.step(action)
          self.add_frame(self.preprocess(new_state))
          new_state = self.stack_frame()
          # if episode % self.render_interval == 0 and episode != 0:
          #     self.env.render()
          state = new_state

    def preprocess(self, state):
        def to_grayscale(img):
            r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
            return 0.2989 * r + 0.587 * g + 0.114 * b
        result = to_grayscale(state)
        result = result[20:-15, 15:-15]
        # plt.imshow(result)
        # plt.show()
        # exit()
        result = result.reshape(*result.shape,1)
        return result
