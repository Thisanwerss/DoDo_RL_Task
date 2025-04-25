# evaluate.py

import time
import argparse
from src.environment import create_env
from src.agent import QLearningAgent
from config import CONFIG

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained CartPole agent')
    parser.add_argument(
        '--model', 
        type=str, 
        default=CONFIG["model_path"],
        help='Path to the model file (default: config model_path)'
    )
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=10, 
        help='Number of episodes to evaluate (default: 10)'
    )
    return parser.parse_args()

def evaluate_agent(model_path, num_episodes=10):
    """
    Evaluate the trained Q-learning agent on the CartPole environment.
    
    Args:
        model_path (str): Path to the trained model file
        num_episodes (int): Number of episodes to evaluate
        
    Returns:
        float: Average reward across all episodes
    """
    print(f"\n🔍 Evaluating model from: {model_path}")
    print(f"🎮 Running for {num_episodes} episodes...\n")
    
    # Create the environment with human rendering
    env = create_env(CONFIG["env_name"], render_mode="human")

    # Initialize the agent
    agent = QLearningAgent(env, CONFIG)

    # Load the trained Q-table
    agent.load(model_path)

    # Evaluate the agent
    start_time = time.time()
    avg_reward = agent.evaluate(episodes=num_episodes)
    eval_time = time.time() - start_time
    
    # Print evaluation summary
    print(f"\n⏱️  Evaluation time: {eval_time:.2f} seconds")
    print(f"🏆 Average reward: {avg_reward:.2f}")
    
    if avg_reward >= 475:
        print("🌟 Excellent performance! Your agent has mastered CartPole!")
    elif avg_reward >= 350:
        print("👍 Good performance! Your agent is learning well.")
    else:
        print("🤔 There's room for improvement. Try adjusting your training parameters.")
        
    return avg_reward

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_agent(args.model, args.episodes)