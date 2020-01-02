from agent import Agent
import gym
import json
import matplotlib
import numpy as np
import subprocess as sh


CONFIG_FILE = 'config.json'

def commit():
	sh.run('git add .', check=True, shell=True)
	sh.run('git commit -m "Training..."', check=True, shell=True)
	sh.run('git push origin master', check=True, shell=True)


def main():
	env = gym.make('SpaceInvaders-ram-v0')
	games = 1000
	save_interval = 10
	agent = Agent(CONFIG_FILE, input_shape=128, n_actions=6)
	scores = []

	for i in range(games):
		done = False
		score = 0
		from_state = env.reset()
		current_lives = 3
		while not done:
			action = agent.choose_action(from_state)
			to_state, reward, done, info = env.step(action)
			score += reward
			if done and info['ale.lives'] == 0:
				reward = -1e5
			elif info['ale.lives'] < current_lives:
				reward = -100
				current_lives = info['ale.lives']
			elif reward == 0:
				reward = -1
			agent.buffer.store(from_state, action, reward, to_state, done)
			from_state = to_state
			agent.train()
			#env.render()
		scores.append(score)
		avg_score = np.mean(scores[max(0, i-10): (i+1)])
		if i % save_interval == 0 and i != 0:
			agent.save(i)
			commit()
		print(f'episode:{i} score:{round(score,2)} avg_score:{round(avg_score,2)}')



if __name__ == '__main__':
    main()