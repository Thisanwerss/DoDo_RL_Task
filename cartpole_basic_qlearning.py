import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make('CartPole-v1')

NUM_BUCKETS = (24, 24, 48, 48)  
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-3.0, 3.0]  
state_bounds[3] = [-3.0, 3.0]  

def discretize_state(state):
    """Convert continuous state to discrete state buckets"""
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

q_table = np.zeros(NUM_BUCKETS + (env.action_space.n,))

alpha = 0.1            # Learning rate
gamma = 0.99           # Discount factor
epsilon = 1.0          # Exploration rate
epsilon_min = 0.01     # Minimum exploration rate
epsilon_decay = 0.990  # Decay rate for exploration

episodes = 5000        
goal_average_reward = 195  
consecutive_episodes = 100 
rewards_per_episode = []
moving_avg_rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()  
        else:
            action = np.argmax(q_table[state])  

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = discretize_state(next_state)

        best_future_q = np.max(q_table[next_state])
        q_table[state + (action,)] = (1 - alpha) * q_table[state + (action,)] + alpha * (reward + gamma * best_future_q)

        state = next_state
        total_reward += reward

    rewards_per_episode.append(total_reward)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    if episode >= consecutive_episodes - 1:
        moving_avg = np.mean(rewards_per_episode[-consecutive_episodes:])
        moving_avg_rewards.append(moving_avg)
        if moving_avg >= goal_average_reward:
            print(f"\nEnvironment solved in {episode+1} episodes! 🎯")
            print(f"Average reward over last {consecutive_episodes} episodes: {moving_avg:.2f}")
            break
    else:
        moving_avg_rewards.append(np.mean(rewards_per_episode[:episode+1]))
    if (episode + 1) % 50 == 0:
        if episode >= consecutive_episodes - 1:
            moving_avg = np.mean(rewards_per_episode[-consecutive_episodes:])
            print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.4f}, Avg Reward = {moving_avg:.2f}")
        else:
            print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.4f}")
env.close()

np.save('q_table.npy', q_table)
discretization_info = {
    'num_buckets': NUM_BUCKETS,
    'state_bounds': state_bounds
}
np.save('q_table.npy', discretization_info)
print("Q-table and discretization info saved successfully!")

plt.figure(figsize=(12, 6))
plt.plot(rewards_per_episode, alpha=0.5, label='Episode Rewards')
plt.plot(moving_avg_rewards, linewidth=2, label=f'Moving Average ({consecutive_episodes} episodes)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning Progress on CartPole')
plt.legend()
plt.grid(True)
plt.savefig('training_progress.png')
plt.show()