# checkpoint_evaluator.py

import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from src.environment import create_env
from src.agent import QLearningAgent
from config import CONFIG

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare CartPole models at different training stages')
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Run in quiet mode without rendering'
    )
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=10, 
        help='Number of episodes to evaluate each model (default: 10)'
    )
    return parser.parse_args()

def evaluate_model(model_path, title, render=True, episodes=10):
    """
    Evaluate a specific model and show its performance.
    
    Args:
        model_path (str): Path to the saved model file
        title (str): Title to display during evaluation
        render (bool): Whether to render the environment
        episodes (int): Number of episodes to evaluate
        
    Returns:
        tuple: (average_reward, rewards_list)
    """
    print(f"\n{'='*50}")
    print(f"Evaluating {title}")
    print(f"{'='*50}")
    
    # Create the environment with appropriate rendering
    render_mode = "human" if render else None
    env = create_env(CONFIG["env_name"], render_mode=render_mode)
    
    # Initialize the agent
    agent = QLearningAgent(env, CONFIG)
    
    # Load the specified model
    agent.load(model_path)
    
    # Evaluate the agent
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = agent.discretize_state(state)
        done = False
        total_reward = 0
        
        while not done:
            action = np.argmax(agent.q_table[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = agent.discretize_state(next_state)
            total_reward += reward
            
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")
    
    avg_reward = sum(rewards) / len(rewards)
    print(f"\nAverage Reward: {avg_reward:.2f}")
    
    if render:
        print("\nPress Enter to continue to the next model...")
        input()
    
    return avg_reward, rewards

def find_checkpoint_files():
    """Find all checkpoint files in the checkpoint directory."""
    checkpoint_dir = CONFIG["checkpoint_dir"]
    checkpoints = []
    
    # First check if the directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory {checkpoint_dir} not found.")
        return checkpoints
    
    # Look for checkpoint files
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("q_table_") and filename.endswith("percent.pkl"):
            # Extract percentage from filename
            try:
                percentage = int(filename.replace("q_table_", "").replace("percent.pkl", ""))
                checkpoints.append((percentage, os.path.join(checkpoint_dir, filename)))
            except ValueError:
                continue
    
    # Sort by percentage
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def plot_comparison_results(results):
    """
    Plot comparison results.
    
    Args:
        results (list): List of (title, avg_reward, rewards) tuples
    """
    # Extract data for plotting
    titles = [r[0] for r in results]
    avg_rewards = [r[1] for r in results]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot average rewards
    ax1.bar(titles, avg_rewards, color='skyblue')
    ax1.set_title('Average Reward by Training Stage')
    ax1.set_xlabel('Training Stage')
    ax1.set_ylabel('Average Reward')
    ax1.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot reward distributions as box plots
    all_rewards = [r[2] for r in results]
    ax2.boxplot(all_rewards, labels=titles)
    ax2.set_title('Reward Distribution by Training Stage')
    ax2.set_xlabel('Training Stage')
    ax2.set_ylabel('Reward')
    ax2.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\n📊 Comparison plot saved to 'model_comparison.png'")
    plt.close()

def main():
    """Main function to compare models."""
    args = parse_arguments()
    
    print("\n🚀 Starting CartPole Model Comparison")
    
    # Find checkpoint files
    checkpoints = find_checkpoint_files()
    
    # Add final model if available
    if os.path.exists(CONFIG["model_path"]):
        checkpoints.append((100, CONFIG["model_path"]))
    
    if not checkpoints:
        print("❌ No checkpoint files found. Run train.py first to create checkpoints.")
        return
    
    print(f"\nFound {len(checkpoints)} models to compare:")
    for percentage, path in checkpoints:
        print(f"  - {percentage}% Training: {path}")
    
    # Define models to compare
    models = [(path, f"Agent at {percentage}% Training") for percentage, path in checkpoints]
    
    # Evaluate each model
    results = []
    for model_path, title in models:
        avg_reward, rewards = evaluate_model(
            model_path, 
            title, 
            render=not args.quiet,
            episodes=args.episodes
        )
        results.append((title, avg_reward, rewards))
    
    # Print summary
    print("\n📊 Model Comparison Summary:")
    print(f"{'Model':30} | {'Average Reward':15}")
    print(f"{'-'*30} | {'-'*15}")
    for title, avg_reward, _ in results:
        print(f"{title:30} | {avg_reward:15.2f}")
    
    # Plot results
    plot_comparison_results(results)
    
    print("\n🎉 Comparison complete!")

if __name__ == "__main__":
    main()