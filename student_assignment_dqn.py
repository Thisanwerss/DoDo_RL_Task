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

        # Detect device: use GPU if available, otherwise fall back to CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
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
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

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

# --- TODO: Custom Reward Function ---
def calculate_custom_reward(state, action, original_reward, done):
    """
    Design your own reward function here!
    
    Args:
        state (np.array): The current state vector [cart_pos, cart_vel, pole_angle, pole_vel]
        action (int): The action taken (0: Left, 1: Right)
        original_reward (float): The default reward from Gym (+1 for every step alive)
        done (bool): Whether the episode has ended (pole fell or out of bounds)
        
    Returns:
        float: The modified reward value
        
    Hints:
        - state[0]: Cart Position (Range: -4.8 to 4.8, termination at ±2.4)
        - state[1]: Cart Velocity
        - state[2]: Pole Angle (Range: ±0.418 rad, termination at ±0.2095 rad)
        - state[3]: Pole Angular Velocity
        
    Goals:
        1. Keep the pole upright (implied by survival).
        2. Keep the cart close to the center (state[0] close to 0).
        3. Make the movement smooth (minimize large velocities if possible).
        
    Challenge:
        - Try to penalize the agent if it moves too far from the center.
        - Try to penalize high angular velocity to make it stable.
    """
    
    # --- YOUR CODE HERE ---
    
    # Example Structure (Dummy components):
    # 1. Survival Reward (Keep it alive)
    r_survival = original_reward 
    
    # 2. Position Reward (Dummy placeholder - currently does nothing)
    # Hint: Maybe 1.0 - abs(state[0]) / 2.4 ?
    r_position = 0.0 
    
    # 3. Stability Reward (Dummy placeholder - currently does nothing)
    # Hint: Maybe -abs(state[3]) ?
    r_stability = 0.0
    
    # Combine them with weights (Tune these weights!)
    w1 = 1.0
    w2 = 0.0
    w3 = 0.0
    
    modified_reward = w1 * r_survival + w2 * r_position + w3 * r_stability
    
    # Default is just survival (Agent learns basics but wobbles)
    # Change this to 0.0 if you want to force yourself to write code immediately!
    # modified_reward = 0.0 
    
    return modified_reward
    # ----------------------


# --- Main Training Loop ---
env = gymnasium.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

episodes = 2000
rewards_per_episode = []

print("Starting training...")

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    step = 0
    while not done:
        step += 1
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Determine if episode is done
        done = terminated or truncated
        
        # Calculate custom reward
        modified_reward = calculate_custom_reward(next_state, action, reward, done)
        
        agent.remember(state, action, modified_reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward  # We track the ORIGINAL reward to see how long it survives

    rewards_per_episode.append(total_reward)

    # Update target network every 10 episodes
    if (episode+1) % 10 == 0:
        agent.update_target_model()

    if (episode+1) % 10 == 0:
        print(f"Episode {episode+1}: Original Reward = {total_reward:.2f}, Steps = {step}, Epsilon = {agent.epsilon:.4f}")

env.close()

# Save the model
torch.save(agent.model.state_dict(), "student_model.pth")
print("Model saved as 'student_model.pth'!")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('My DQN Training Performance')
plt.grid()
plt.savefig('student_training_curve.png')
print("Training curve saved as 'student_training_curve.png'.")
