# config.py

# Training Configuration for Q-Learning Agent on CartPole

CONFIG = {
    # Name of the environment
    "env_name": "CartPole-v1",

    # Total number of training episodes
    "episodes": 5000,

    # Learning rate (α): reduced to slow down learning
    "alpha": 0.05,  # Reduced from 0.1

    # Discount factor (γ): importance of future rewards
    "gamma": 0.95,  # Slightly reduced from 0.99

    # Exploration rate (ε): increased initial exploration
    "epsilon": 1.0,

    # Minimum exploration rate (ε_min): increased to maintain more exploration
    "epsilon_min": 0.1,  # Increased from 0.01

    # Decay rate for ε per episode: slower decay
    "epsilon_decay": 0.998,  # Slower decay from 0.995

    # Save model path
    "model_path": "models/q_table.pkl"
}