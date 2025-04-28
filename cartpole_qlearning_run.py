import gymnasium
import numpy as np
import time

env = gymnasium.make('CartPole-v1', render_mode="human")
q_table = np.load('q_table.npy')

discretization_info = np.load('discretization_info.npy', allow_pickle=True).item()
NUM_BUCKETS = discretization_info['num_buckets']
state_bounds = discretization_info['state_bounds']

def discretize_state(state):
    """Convert continuous state to discrete state buckets - same as in training"""
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

episodes = 100
for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0
    step = 0
    
    print(f"\nEpisode {episode + 1} starting...")
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        state = discretize_state(next_state)
        total_reward += reward
        step += 1
        time.sleep(0.01)
    
    print(f"Episode {episode + 1} finished: Steps = {step}, Total Reward = {total_reward}")

env.close()