import gymnasium
import numpy as np
import matplotlib.pyplot as plt

# Create environment
env = gymnasium.make('CartPole-v1')

# --- Discretization Settings (Improved) ---
NUM_BUCKETS = (24, 24, 48, 48)  # More fine-grained buckets

state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-3.0, 3.0]   # clip cart velocity
state_bounds[3] = [-3.0, 3.0]   # clip pole angular velocity

def discretize_state(state):
    ratios = []
    for i in range(len(state)):
        low, high = state_bounds[i]
        if state[i] <= low:
            ratios.append(0.0)
        elif state[i] >= high:
            ratios.append(1.0)
        else:
            ratios.append((state[i] - low) / (high - low))

    new_state = []
    for i in range(len(ratios)):
        bucket_index = int(round((NUM_BUCKETS[i] - 1) * ratios[i]))
        bucket_index = min(NUM_BUCKETS[i] - 1, max(0, bucket_index))
        new_state.append(bucket_index)

    return tuple(new_state)

# --- Initialize Q-table ---
q_table = np.zeros(NUM_BUCKETS + (env.action_space.n,))

# --- Hyperparameters ---
alpha = 0.1            # learning rate
gamma = 0.99           # discount factor
epsilon = 1.0          # exploration rate
epsilon_min = 0.01     # minimum exploration rate
epsilon_decay = 0.995  # smoother decay
episodes = 1000        # train more episodes
goal_average_reward = 195  # solve CartPole if average reward > 195
consecutive_episodes = 100

# For tracking
rewards_per_episode = []
moving_avg_rewards = []

# --- Training Loop ---
for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        # --- Choose Action ---
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(q_table[state])  # exploit

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_state = discretize_state(next_state)

        # --- Update Q-table ---
        best_future_q = np.max(q_table[next_state])
        q_table[state + (action,)] = (1 - alpha) * q_table[state + (action,)] + alpha * (reward + gamma * best_future_q)

        state = next_state
        total_reward += reward

    rewards_per_episode.append(total_reward)

    # Update epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Calculate Moving Average
    if episode >= consecutive_episodes:
        moving_avg = np.mean(rewards_per_episode[-consecutive_episodes:])
        moving_avg_rewards.append(moving_avg)

        if moving_avg >= goal_average_reward:
            print(f"\nEnvironment solved in {episode+1} episodes! 🎯")
            break
    else:
        moving_avg_rewards.append(np.mean(rewards_per_episode))

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.4f}")

env.close()

# --- Save the Q-table ---
np.save('q_table.npy', q_table)
print("Q-table saved successfully!")

# --- Plotting ---
plt.plot(rewards_per_episode, label='Rewards')
plt.plot(moving_avg_rewards, label='Moving Average (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Improved Q-Learning on CartPole')
plt.legend()
plt.grid()
plt.show()
