import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --- Neural Network ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.memory = deque(maxlen=10000)   # Replay buffer
        self.batch_size = 64

        self.gamma = 0.99

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute current Q values
        curr_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        next_q = self.target_model(next_states).max(1)[0]
        expected_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(curr_q, expected_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Main Training with Improved Reward ---
env = gymnasium.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

episodes = 2000
rewards_per_episode = []

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    step = 0
    while not done:
        step += 1
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Add position penalty to encourage staying centered
        cart_position = next_state[0]
        position_penalty = abs(cart_position) * 0.1  # Scale position penalty
        modified_reward = reward - position_penalty
        
        # Also end early if the cart moves too far
        position_done = abs(cart_position) > 2.4  # Cart position threshold
        done = terminated or truncated or position_done
        
        agent.remember(state, action, modified_reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward  # Track original reward for metrics

    rewards_per_episode.append(total_reward)

    # Update target network every 10 episodes
    if (episode+1) % 10 == 0:
        agent.update_target_model()

    if (episode+1) % 10 == 0:
        print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Steps = {step}, Epsilon = {agent.epsilon:.4f}")

env.close()

# Save the model
torch.save(agent.model.state_dict(), "dqn_cartpole_centered.pth")
print("Model saved successfully!")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Deep Q-Learning on CartPole with Position Constraints')
plt.grid()
plt.savefig('dqn_cartpole_training_curve.png')
plt.show()