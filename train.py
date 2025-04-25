# train.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from src.environment import create_env
from src.agent import QLearningAgent
from config import CONFIG

def create_checkpoints_dir():
    """Create directory for saving model checkpoints."""
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    print(f"📁 Checkpoint directory created: {CONFIG['checkpoint_dir']}")

def save_checkpoint(agent, episode, total_episodes):
    """
    Save agent checkpoint at specified episode.
    
    Args:
        agent: The Q-learning agent
        episode: Current episode number
        total_episodes: Total number of episodes for training
    """
    percentage = (episode + 1) / total_episodes
    checkpoint_path = os.path.join(
        CONFIG["checkpoint_dir"], 
        f"q_table_{int(percentage * 100)}percent.pkl"
    )
    agent.save(checkpoint_path)
    print(f"🔖 Saved checkpoint at {int(percentage * 100)}% of training")

def plot_rewards(rewards, window_size=100, save_path="training_rewards.png"):
    """
    Plot the episode rewards to visualize training progress.
    
    Args:
        rewards (list): List of rewards for each episode
        window_size (int): Size of moving average window
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot moving average if we have enough data
    if len(rewards) >= window_size:
        # Calculate moving average
        moving_avg = np.convolve(
            rewards, np.ones(window_size)/window_size, mode='valid'
        )
        plt.plot(
            range(window_size-1, len(rewards)), 
            moving_avg, 
            color='red', 
            label=f'{window_size}-Episode Moving Average'
        )
    
    plt.title("CartPole Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"📊 Training progress plot saved to {save_path}")
    plt.close()

def main():
    """Main training function."""
    print("\n🚀 Starting CartPole Q-Learning Training\n")
    
    # Create directories for saving models and checkpoints
    os.makedirs("models", exist_ok=True)
    create_checkpoints_dir()
    
    # Step 1: Create the environment
    env = create_env(CONFIG["env_name"])
    
    # Step 2: Initialize the Q-Learning Agent
    agent = QLearningAgent(env, CONFIG)
    
    # Step 3: Calculate checkpoint episodes
    total_episodes = CONFIG["episodes"]
    checkpoint_episodes = [
        int(p * total_episodes) - 1 for p in CONFIG["checkpoint_percentages"]
    ]
    
    # Step 4: Start training
    start_time = time.time()
    
    # Train and collect rewards
    rewards = []
    for episode in range(total_episodes):
        # Train for one episode
        state, _ = env.reset()
        state_discrete = agent.discretize_state(state)
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.choose_action(state_discrete)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_discrete = agent.discretize_state(next_state)
            
            # Store experience in replay buffer
            agent.replay_buffer.append((state_discrete, action, reward, next_state_discrete, done))
            
            # Learn from experiences
            if len(agent.replay_buffer) >= agent.replay_start_size:
                agent.learn_from_replay()
                
            state_discrete = next_state_discrete
            episode_reward += reward
            
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record reward
        rewards.append(episode_reward)
        
        # Save checkpoint if needed
        if episode in checkpoint_episodes:
            save_checkpoint(agent, episode, total_episodes)
            
        # Periodically plot rewards (every 200 episodes)
        if (episode + 1) % 200 == 0 or episode == total_episodes - 1:
            plot_rewards(rewards)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Step 5: Save final model
    agent.save(CONFIG["model_path"])
    
    # Plot final rewards
    plot_rewards(rewards)
    
    # Print training summary
    print("\n✅ Training completed!")
    print(f"⏱️  Training time: {training_time:.2f} seconds")
    print(f"📈 Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"📊 Peak average reward: {np.max([np.mean(rewards[i:i+100]) for i in range(len(rewards)-100)]):.2f}")
    print("\n🎮 You can now run 'evaluate.py' to see how your agent performs!")
    print("🔍 Or run 'checkpoint_evaluator.py' to compare models at different training stages.")

if __name__ == "__main__":
    main()