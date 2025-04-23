# utils.py

import numpy as np

def create_discretization_bounds(env, buckets=(1, 1, 6, 12)):
    """
    Creates the bounds for state discretization in the Q-learning agent.

    Args:
        env: Gym environment
        buckets: Number of buckets to divide the state space into
    """
    upper_bounds = [
        env.observation_space.high[0],  # Position
        0.5,                             # Velocity
        env.observation_space.high[2],  # Angle
        np.radians(50)                  # Angular velocity
    ]
    lower_bounds = [
        env.observation_space.low[0],   # Position
        -0.5,                            # Velocity
        env.observation_space.low[2],   # Angle
        -np.radians(50)                 # Angular velocity
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
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i])
              for i in range(len(state))]

    new_state = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]

    return tuple(new_state)
