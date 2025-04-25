# config.py

# Training Configuration for Q-Learning Agent on CartPole

CONFIG = {
    # Name of the environment
    "env_name": "CartPole-v1",

    # Total number of training episodes
    "episodes": 2000,  # Reduced from 5000 for faster training

    # Learning rate (α): how quickly the agent updates its Q-values
    "alpha": 0.1,  

    # Discount factor (γ): importance of future rewards (0-1)
    "gamma": 0.99,  # Higher value makes agent more focused on long-term rewards

    # Exploration rate (ε): probability of taking a random action
    "epsilon": 1.0,  # Start with 100% exploration

    # Minimum exploration rate (ε_min): lowest exploration probability
    "epsilon_min": 0.05,  

    # Decay rate for ε per episode: controls exploration decrease
    "epsilon_decay": 0.995,  # Adjusted for better learning curve

    # Save model path
    "model_path": "models/q_table.pkl",
    
    # Checkpoint directory
    "checkpoint_dir": "checkpoints",
    
    # Save checkpoints at these percentages of training
    "checkpoint_percentages": [0.1, 0.25, 0.5, 0.75, 1.0]
}