from agent import Agent
import gym
import json
import matplotlib
import numpy as np

CONFIG_FILE = 'config.json'

def fill_buffer(agent, env):
	print("Filling buffer...")
	while agent.buffer.count < agent.buffer.size:
		done = False
		from_state = env.reset()
		while not done:
			action = env.action_space.sample()
			to_state, reward, done, info = env.step(action)
			if done and info['ale.lives'] == 0:
					reward = -100
			agent.buffer.store(from_state, action, reward, to_state, done)
			from_state = to_state
	print("Buffer filled!")



def main():
	env = gym.make('SpaceInvaders-ram-v0')
	games = 1000
	agent = Agent(CONFIG_FILE, input_shape=128, n_actions=6)
	scores = []

	fill_buffer(agent, env)

	for i in range(games):
		done = False
		score = 0
		from_state = env.reset()
		while not done:
			action = agent.choose_action(from_state)
			to_state, reward, done, info = env.step(action)
			score += reward
			if done and info['ale.lives'] == 0:
				reward = -100
			agent.buffer.store(from_state, action, reward, to_state, done)
			from_state = to_state
			agent.train()
			#env.render()
		scores.append(score)
		avg_score = np.mean(scores[max(0, i-10): (i+1)])
		print(f'episode:{i} score{round(score,2)} avg_score:{round(avg_score,2)}')



if __name__ == '__main__':
    main()