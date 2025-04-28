# CartPole-v1 Reinforcement Learning Project

This repository contains implementations of different reinforcement learning algorithms to solve the classic CartPole-v1 environment from OpenAI Gymnasium. The project demonstrates the progression from simple rule-based approaches to more sophisticated deep reinforcement learning techniques.

![CartPole Environment](https://gymnasium.farama.org/_images/cart_pole.gif)
*Image source: Gymnasium documentation*

## 🎯 Project Overview

The CartPole-v1 environment features a pole balanced on a cart that moves left or right. The goal is to keep the pole upright for as long as possible. The environment is considered "solved" when the agent achieves an average reward of 195 or more over 100 consecutive episodes.

This project implements three approaches of increasing complexity:
1. Simple rule-based agent
2. Q-learning with state discretization
3. Deep Q-Network (DQN)

## 📋 Contents

- `cartpole_basic.py` - Simple rule-based agent that decides actions based on pole angle
- `cartpole_basic_qlearning.py` - Q-learning implementation with state discretization
- `cartpole_dqn.py` - Deep Q-Network implementation using PyTorch
- `cartpole_dqn_run.py` - Script to run and visualize the trained DQN model
- `cartpole_qlearning_run.py` - Script to run and visualize the trained Q-learning model

## 🤖 Implemented Algorithms

### 1. Rule-Based Agent
A simple baseline approach that decides actions based solely on the pole's angle:
- If the pole is leaning left (angle < 0), move the cart left
- If the pole is leaning right (angle > 0), move the cart right

### 2. Q-Learning
An improvement over the rule-based approach, using:
- State discretization to handle continuous state space
- Exploration vs. exploitation with epsilon-greedy policy
- Experience replay to improve learning
- Hyperparameter tuning for optimal performance

### 3. Deep Q-Network (DQN)
A more advanced approach using neural networks:
- Neural network to approximate the Q-function
- Experience replay buffer for stable learning
- Target network for more stable updates
- Position constraints to encourage centered cart position
- Epsilon-greedy exploration strategy

## 🔧 Installation and Requirements

1. Clone this repository:
```bash
git clone https://github.com/yourusername/cartpole-rl.git
cd cartpole-rl
```

2. Install required packages:
```bash
pip install gymnasium numpy matplotlib torch
```

## 🚀 Usage

### Training the agents

1. Run the rule-based agent:
```bash
python cartpole_basic.py
```

2. Train the Q-learning agent:
```bash
python cartpole_basic_qlearning.py
```

3. Train the DQN agent:
```bash
python cartpole_dqn.py
```

### Visualizing trained agents

1. To visualize the Q-learning agent:
```bash
python cartpole_qlearning_run.py
```

2. To visualize the DQN agent:
```bash
python cartpole_dqn_run.py
```

## 📊 Results

### Performance Comparison

| Algorithm | Average Reward | Episodes to Solve |
|-----------|----------------|------------------|
| Rule-Based | ~20-50         | Does not solve   |
| Q-Learning | ~150-200       | ~800-1000        |
| DQN        | 200+           | ~500-700         |

The DQN implementation outperforms both the rule-based and Q-learning approaches, achieving the target performance more consistently and in fewer episodes.

### Key Insights

1. **State Representation**: Finer discretization in Q-learning significantly improves performance
2. **Exploration Strategy**: Properly decaying epsilon leads to better policy convergence
3. **Position Constraints**: Adding a position penalty in DQN helps keep the cart centered

## 📝 Project Structure

```
cartpole-rl/
├── cartpole_basic.py
├── cartpole_basic_qlearning.py
├── cartpole_dqn.py
├── cartpole_dqn_run.py
├── cartpole_qlearning_run.py
├── q_table.npy
├── dqn_cartpole_centered.pth
└── README.md
```

## 🔄 Future Improvements

- Implement additional algorithms like DDQN, A3C, or PPO
- Experiment with different neural network architectures
- Apply these techniques to more complex environments
- Add visualization of training progress
- Optimize hyperparameters using automated methods

## 📚 Resources

- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [Sutton & Barto: Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html)
- [Deep Reinforcement Learning Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.