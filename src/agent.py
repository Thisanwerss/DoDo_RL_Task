# agent.py

import numpy as np
import os
import pickle
import random
from collections import deque
from tqdm import tqdm  # Progress bar for training visualization

class QLearningAgent:
    def __init__(self, env, config):
        """
        Initialize the Q-learning agent with experience replay.
        
        Args:
            env: Gymnasium environment
            config: Dictionary of configuration parameters
        """
        self.env = env
        self.config = config

        # Get action space (CartPole has 2 actions: left=0, right=1)
        self.action_space = env.action_space.n

        # Discretization settings - improved for better state representation
        self.buckets = (3, 3, 6, 6)  # Increased buckets for position and velocity

        # Q-table has dimensions = buckets + action space
        self.q_table = np.zeros(self.buckets + (self.action_space,))

        # Set bounds for discretization
        self.upper_bounds = [
            self.env.observation_space.high[0],  # Position
            0.5,                                 # Velocity
            self.env.observation_space.high[2],  # Angle
            np.radians(50)                       # Angular velocity
        ]
        self.lower_bounds = [
            self.env.observation_space.low[0],   # Position
            -0.5,                                # Velocity
            self.env.observation_space.low[2],   # Angle
            -np.radians(50)                      # Angular velocity
        ]
        
        # Experience replay buffer - improves learning stability
        self.replay_buffer_size = 10000  
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.batch_size = 64  # Increased from 32 for better learning
        self.replay_start_size = 500  # Minimum experiences before training
        
        # Initialize epsilon for exploration
        self.epsilon = config["epsilon"]

    def train_for_episodes(self, num_episodes):
        """
        Train the agent for a specific number of episodes with experience replay.
        
        Args:
            num_episodes: Number of episodes to train for
            
        Returns:
            list: Episode rewards
        """
        rewards = []
        # Track best average reward to detect improvement
        best_avg_reward = -float('inf')
        
        # Progress bar for training
        progress_bar = tqdm(range(num_episodes), desc="Training")
        
        for episode in progress_bar:
            state, _ = self.env.reset()
            state_discrete = self.discretize_state(state)
            episode_reward = 0
            steps = 0

            done = False
            while not done:
                # Choose action using epsilon-greedy policy
                action = self.choose_action(state_discrete)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state_discrete = self.discretize_state(next_state)
                
                # Store experience in replay buffer
                self.replay_buffer.append((state_discrete, action, reward, next_state_discrete, done))
                
                # Learn from experiences when buffer has enough samples
                if len(self.replay_buffer) >= self.replay_start_size:
                    self.learn_from_replay()

                state_discrete = next_state_discrete
                episode_reward += reward
                steps += 1

            # Decay epsilon after each episode
            self.decay_epsilon()
            
            rewards.append(episode_reward)
            
            # Update progress bar with useful information
            # Calculate moving average of last 100 episodes
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                
            progress_bar.set_postfix({
                'reward': episode_reward, 
                'avg_reward': f'{avg_reward:.1f}',
                'epsilon': f'{self.epsilon:.3f}',
                'best_avg': f'{best_avg_reward:.1f}'
            })

        return rewards
        
    def learn_from_replay(self):
        """
        Sample from replay buffer and update Q-values for improved learning.
        """
        # Sample random batch from replay buffer
        batch_size = min(len(self.replay_buffer), self.batch_size)
        batch = random.sample(self.replay_buffer, batch_size)
        
        for state, action, reward, next_state, done in batch:
            # Calculate Q-value target using Q-learning update rule
            if done:
                target = reward
            else:
                target = reward + self.config["gamma"] * np.max(self.q_table[next_state])
            
            # Update Q-value with learning rate alpha
            current = self.q_table[state][action]
            self.q_table[state][action] = current + self.config["alpha"] * (target - current)

    def discretize_state(self, state):
        """
        Convert a continuous state to a discrete state.

        Args:
            state (array): The state from the environment

        Returns:
            tuple: Discretized state
        """
        ratios = []
        for i in range(len(state)):
            bound_range = self.upper_bounds[i] - self.lower_bounds[i]
            # Avoid division by zero
            if bound_range == 0:
                ratios.append(0)
            else:
                adjusted_value = state[i] + abs(self.lower_bounds[i])
                ratios.append(adjusted_value / bound_range)

        # Clip values to ensure they're within bounds
        new_state = []
        for i in range(len(state)):
            bucket_idx = int(round((self.buckets[i] - 1) * ratios[i]))
            new_state.append(min(self.buckets[i] - 1, max(0, bucket_idx)))

        return tuple(new_state)

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy strategy.

        Args:
            state (tuple): Discretized state

        Returns:
            int: Chosen action
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore - random action
        else:
            return np.argmax(self.q_table[state])  # Exploit - best action

    def decay_epsilon(self):
        """Decay epsilon according to the configured decay rate"""
        if self.epsilon > self.config["epsilon_min"]:
            self.epsilon *= self.config["epsilon_decay"]
    
    def train(self):
        """
        Main training loop for all episodes.
        
        Returns:
            list: Episode rewards
        """
        return self.train_for_episodes(self.config["episodes"])

    def evaluate(self, render=False, episodes=5):
        """
        Evaluate the trained model.

        Args:
            render (bool): Whether to visually render the environment
            episodes (int): Number of episodes to evaluate
            
        Returns:
            float: Average reward across evaluation episodes
        """
        total_rewards = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                # Always choose best action during evaluation
                action = np.argmax(self.q_table[state])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = self.discretize_state(next_state)
                episode_reward += reward
                steps += 1

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {steps}")

        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.2f}")
        return avg_reward

    def save(self, filename):
        """
        Save the Q-table to disk.
        
        Args:
            filename (str): Path to save the Q-table
        """
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                pickle.dump(self.q_table, f)
            print(f"✅ Q-table saved to {filename}")
        except Exception as e:
            print(f"⚠️ Error saving Q-table: {e}")

    def load(self, filename):
        """
        Load the Q-table from disk.
        
        Args:
            filename (str): Path to load the Q-table from
        """
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"📂 Q-table loaded from {filename}")
        except FileNotFoundError:
            print(f"⚠️ File not found: {filename}")
        except Exception as e:
            print(f"⚠️ Error loading Q-table: {e}")