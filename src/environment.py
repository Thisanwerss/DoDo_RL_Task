# environment.py

import gymnasium as gym

def create_env(env_name: str, render_mode: str = None):
    """
    Create and return the Gymnasium environment.

    Args:
        env_name (str): The name of the environment (e.g., "CartPole-v1").
        render_mode (str, optional): Use "human" if you want to see the environment visually during evaluation.

    Returns:
        env (gym.Env): The initialized environment.
    """

    # Initialize the Gym environment using the given name.
    # If render_mode="human", it will open a window to visualize the agent's actions.
    env = gym.make(env_name, render_mode=render_mode)

    return env
