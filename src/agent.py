# agent.py

import numpy as np
import os
import pickle  # Used to save/load Q-table
from tqdm import tqdm  # Nice progress bar for training visualization

class QLearningAgent:
    def __init__(self, env, config):
        """
        Initialize the Q-learning agent.

        Args:
            env: Gym environment
            config: Dictionary of training configurations
        """
        self.env = env
        self.config = config

        # Get action space (CartPole has 2 actions: left=0, right=1)
        self.action_space = env.action_space.n

        # CartPole observations are continuous, so we discretize them
        self.buckets = (1, 1, 6, 12)  # Number of buckets for each dimension of state

        # Q-table has dimensions = buckets + action space
        self.q_table = np.zeros(self.buckets + (self.action_space,))

        # Set bounds for discretization
        self.upper_bounds = [
            self.env.observation_space.high[0],
            0.5,
            self.env.observation_space.high[2],
            np.radians(50)
        ]
        self.lower_bounds = [
            self.env.observation_space.low[0],
            -0.5,
            self.env.observation_space.low[2],
            -np.radians(50)
        ]

    def discretize_state(self, state):
        """
        Convert a continuous state to a discrete state.

        Args:
            state (array): The state from the environment

        Returns:
            tuple: Discretized state
        """
        ratios = [(state[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
                  for i in range(len(state))]

        new_state = [int(round((self.buckets[i] - 1) * ratios[i]))
                     for i in range(len(state))]

        new_state = [min(self.buckets[i] - 1, max(0, new_state[i]))
                     for i in range(len(state))]

        return tuple(new_state)

    def choose_action(self, state, epsilon):
        """
        Choose an action using epsilon-greedy strategy.

        Args:
            state (tuple): Discretized state
            epsilon (float): Exploration rate

        Returns:
            int: Chosen action
        """
        if np.random.random() < epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state, alpha, gamma):
        """
        Apply the Q-learning update rule.

        Q(s, a) ← Q(s, a) + α [r + γ max(Q(s', ·)) − Q(s, a)]

        Args:
            state: current state
            action: taken action
            reward: observed reward
            next_state: state after taking action
            alpha: learning rate
            gamma: discount factor
        """
        best_next_action = np.max(self.q_table[next_state])
        td_target = reward + gamma * best_next_action
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += alpha * td_delta

    def train(self):
        """
        Main training loop.
        Trains the Q-table using episodes of CartPole.
        """
        rewards = []

        for episode in tqdm(range(self.config["episodes"]), desc="Training"):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            total_reward = 0

            done = False
            while not done:
                action = self.choose_action(state, self.config["epsilon"])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state_discrete = self.discretize_state(next_state)

                self.update_q_value(state, action, reward, next_state_discrete,
                                    self.config["alpha"], self.config["gamma"])

                state = next_state_discrete
                total_reward += reward

            # Decay epsilon
            if self.config["epsilon"] > self.config["epsilon_min"]:
                self.config["epsilon"] *= self.config["epsilon_decay"]

            rewards.append(total_reward)

        return rewards

    def evaluate(self, render=False, episodes=5):
        """
        Evaluate the trained model.

        Args:
            render (bool): Whether to visually render the environment
            episodes (int): Number of episodes to evaluate
        """
        for episode in range(episodes):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            done = False
            total_reward = 0

            while not done:
                if render:
                    self.env.render()

                action = np.argmax(self.q_table[state])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = self.discretize_state(next_state)
                total_reward += reward

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    def save(self, filename="models/q_table.pkl"):
        """
        Save the Q-table to disk.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"✅ Q-table saved to {filename}")

    def load(self, filename="models/q_table.pkl"):
        """
        Load the Q-table from disk.
        """
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"📂 Q-table loaded from {filename}")
