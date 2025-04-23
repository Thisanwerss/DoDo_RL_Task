# config.py

# Training Configuration for Q-Learning Agent on CartPole

CONFIG = {
    # Name of the environment
    "env_name": "CartPole-v1",

    # Total number of training episodes
    "episodes": 5000,

    # Learning rate (α): how much new info overrides old info
    "alpha": 0.1,

    # Discount factor (γ): importance of future rewards
    "gamma": 0.99,

    # Exploration rate (ε): chance of random action vs best action
    "epsilon": 1.0,

    # Minimum exploration rate (ε_min): floor limit for epsilon
    "epsilon_min": 0.01,

    # Decay rate for ε per episode
    "epsilon_decay": 0.995,

    # Save model path
    "model_path": "models/q_table.pkl"
}
