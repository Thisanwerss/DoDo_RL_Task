# train.py

from src.environment import create_env
from src.agent import QLearningAgent
from config import CONFIG

import matplotlib.pyplot as plt

def plot_rewards(rewards):
    """
    Plot the episode rewards to visualize training progress.
    """
    plt.plot(rewards)
    plt.title("Training Reward Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()


def main():
    # Step 1: Create the environment
    env = create_env(CONFIG["env_name"])

    # Step 2: Initialize the Q-Learning Agent
    agent = QLearningAgent(env, CONFIG)

    # Step 3: Start training and collect rewards per episode
    rewards = agent.train()

    # Step 4: Save the trained model (Q-table)
    agent.save(CONFIG["model_path"])

    # Step 5: Optionally, show reward plot
    plot_rewards(rewards)

    print("\n🎉 Training complete! You can now run `evaluate.py` to test the agent.")

if __name__ == "__main__":
    main()
