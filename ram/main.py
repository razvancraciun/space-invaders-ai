from agent import Agent
import gym
import json
import matplotlib
import numpy as np

CONFIG_FILE = 'config.json'

def main():
	env = gym.make('SpaceInvaders-ram-v0')
	games = 100
	agent = Agent(CONFIG_FILE, input_shape=128, n_actions=6)
	scores = []

	for i in range(games):
		done = False
		score = 0
		from_state = env.reset()
		while not done:
			action = agent.choose_action(from_state)
			to_state, reward, done, _ = env.step(action)
			score += reward
			agent.buffer.store(from_state, action, reward, to_state, done)
			from_state = to_state
			agent.train()
			#env.render()
		scores.append(score)
		avg_score = np.mean(scores[max(0, i-100): (i+1)])
		print(f'episode:{i} score{round(score,2)} avg_score:{round(avg_score,2)}')



if __name__ == '__main__':
    main()