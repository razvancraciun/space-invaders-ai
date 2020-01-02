import gym
from agent import Agent
import numpy as np
from time import sleep

CONFIG_FILE = 'config.json'

def main():
    env = gym.make('SpaceInvaders-ram-v0')
    agent = Agent(CONFIG_FILE, input_shape=128, n_actions=6)
    agent.load('120')
    from_state = env.reset()
    done = False
    while not done:
        action = np.argmax(agent.model.predict(from_state[np.newaxis, :]))
        to_state, reward, done, info = env.step(action)
        agent.buffer.store(from_state, action, reward, to_state, done)
        from_state = to_state
        env.render()
        sleep(0.03)

    



if __name__ == '__main__':
    main()