from agent import Agent
from tensorflow.keras.models import load_model


def main():
    agent = Agent()
    agent.nn.model = load_model('ckpt7.h5')
    agent.play_episode() 


if __name__ == '__main__':
    main()