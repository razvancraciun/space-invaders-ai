from agent import Agent
import numpy as np

def main():
    agent = Agent()
    agent.train(100)
    agent.nn.model.save('asd.h5')


if __name__ == '__main__':
    main()
