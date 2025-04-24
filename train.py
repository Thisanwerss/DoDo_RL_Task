# train.py

from src.environment import create_env
from src.agent import QLearningAgent
from config import CONFIG

import matplotlib.pyplot as plt
import os

def plot_rewards(rewards):
    """
    Plot the episode rewards to visualize training progress.
    """
    plt.plot(rewards)
    plt.title("Training Reward Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig("training_rewards.png")
    plt.show()


def main():
    # Step 1: Create the environment
    env = create_env(CONFIG["env_name"])

    # Step 2: Initialize the Q-Learning Agent
    agent = QLearningAgent(env, CONFIG)

    # Step 3: Start training and collect rewards per episode
    total_episodes = CONFIG["episodes"]
    early_stage = int(total_episodes * 0.1)  # 10% of total episodes
    
    # Make sure the models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Train first 10% and save early model
    print(f"\nTraining for first {early_stage} episodes (10%)...")
    early_rewards = agent.train_for_episodes(early_stage)
    agent.save("models/q_table_10percent.pkl")
    print(f"✅ 10% trained model saved!")
    
    # Train remaining 90% and save final model
    print(f"\nTraining for remaining {total_episodes - early_stage} episodes...")
    final_rewards = agent.train_for_episodes(total_episodes - early_stage)
    agent.save(CONFIG["model_path"])
    
    # Combine rewards for plotting
    rewards = early_rewards + final_rewards

    # Step 5: Show reward plot
    plot_rewards(rewards)

    print("\n🎉 Training complete! You can now run `compare_models.py` to see the progress.")

if __name__ == "__main__":
    main()