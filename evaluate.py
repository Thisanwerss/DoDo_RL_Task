# evaluate.py

from src.environment import create_env
from src.agent import QLearningAgent
from config import CONFIG

def evaluate_agent():
    """
    Evaluate the trained Q-learning agent on the CartPole environment.
    This will run the agent for a few episodes and show the results.
    """
    # Create the environment
    env = create_env(CONFIG["env_name"], render_mode="human")  # Use 'human' to render the environment

    # Initialize the agent
    agent = QLearningAgent(env, CONFIG)

    # Load the trained Q-table (model)
    agent.load(CONFIG["model_path"])

    # Evaluate the agent for a few episodes
    agent.evaluate(render=True, episodes=100)  # Render the environment during evaluation


if __name__ == "__main__":
    evaluate_agent()
