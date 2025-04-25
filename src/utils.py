# utils.py

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import gymnasium as gym

def create_discretization_bounds(env, buckets=(3, 3, 6, 6)):
    """
    Creates the bounds for state discretization in the Q-learning agent.

    Args:
        env: Gym environment
        buckets: Number of buckets to divide the state space into
        
    Returns:
        tuple: (lower_bounds, upper_bounds, buckets)
    """
    upper_bounds = [
        env.observation_space.high[0],  # Position
        0.5,                            # Velocity (estimated)
        env.observation_space.high[2],  # Angle
        np.radians(50)                  # Angular velocity (estimated)
    ]
    
    lower_bounds = [
        env.observation_space.low[0],   # Position
        -0.5,                           # Velocity (estimated)
        env.observation_space.low[2],   # Angle
        -np.radians(50)                 # Angular velocity (estimated)
    ]
    
    return lower_bounds, upper_bounds, buckets

def discretize_state(state, lower_bounds, upper_bounds, buckets):
    """
    Convert a continuous state to a discrete state using specified bounds.

    Args:
        state (array): The current state
        lower_bounds (list): Lower bounds for each dimension
        upper_bounds (list): Upper bounds for each dimension
        buckets (tuple): Number of buckets for each dimension

    Returns:
        tuple: Discretized state
    """
    ratios = []
    for i in range(len(state)):
        bound_range = upper_bounds[i] - lower_bounds[i]
        # Avoid division by zero
        if bound_range == 0:
            ratios.append(0)
        else:
            adjusted_value = state[i] + abs(lower_bounds[i])
            ratios.append(adjusted_value / bound_range)

    # Clip values to ensure they're within bounds
    new_state = []
    for i in range(len(state)):
        bucket_idx = int(round((buckets[i] - 1) * ratios[i]))
        new_state.append(min(buckets[i] - 1, max(0, bucket_idx)))

    return tuple(new_state)

def save_frames_as_gif(frames, filename="cartpole.gif", path="./videos/"):
    """
    Save rendered frames as a gif.
    
    Args:
        frames (list): List of frames (RGB arrays)
        filename (str): Name of the output file
        path (str): Path to save the gif
    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Set up figure and animation
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
    
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50
    )
    
    # Save animation
    anim.save(os.path.join(path, filename), writer='pillow', fps=30)
    
    print(f"🎬 Animation saved to {os.path.join(path, filename)}")

def record_video(agent, env_name, num_episodes=1, filename="cartpole_video.gif"):
    """
    Record a video of the agent's performance.
    
    Args:
        agent: The trained agent
        env_name (str): Name of the environment
        num_episodes (int): Number of episodes to record
        filename (str): Output filename
        
    Returns:
        list: Rewards for each recorded episode
    """
    # Create environment with rgb_array rendering for video recording
    env = gym.make(env_name,