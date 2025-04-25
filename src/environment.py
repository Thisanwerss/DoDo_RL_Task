# environment.py

import gymnasium as gym

def create_env(env_name: str, render_mode: str = None, max_episode_steps: int = None):
    """
    Create and return the Gymnasium environment.

    Args:
        env_name (str): The name of the environment (e.g., "CartPole-v1").
        render_mode (str, optional): The rendering mode.
            - "human": Open a window to visualize the environment
            - "rgb_array": Return RGB frames for video recording
        max_episode_steps (int, optional): Maximum steps per episode before truncation.
            - If None, use default steps from Gym
            - For CartPole-v1, default is 500 steps

    Returns:
        env (gym.Env): The initialized environment.
    """
    try:
        # Initialize the Gym environment using the given name and render mode
        env = gym.make(env_name, render_mode=render_mode, max_episode_steps=max_episode_steps)
        
        print(f"🎮 Environment '{env_name}' created successfully")
        print(f"   - State space shape: {env.observation_space.shape}")
        print(f"   - Action space: {env.action_space}")
        
        return env
        
    except gym.error.Error as e:
        print(f"⚠️ Error creating environment: {e}")
        print("Make sure the environment name is correct and gymnasium is installed.")
        raise