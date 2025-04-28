import gymnasium
import numpy as np

# Create the environment
env = gymnasium.make('CartPole-v1', render_mode="human")

# Load the saved Q-table
q_table = np.load('q_table.npy')

# --- Discretization Settings ---
state_bins = [
    np.linspace(-4.8, 4.8, 10),      # cart position
    np.linspace(-4, 4, 10),          # cart velocity
    np.linspace(-0.418, 0.418, 10),  # pole angle
    np.linspace(-4, 4, 10)           # pole angular velocity
]

# --- Helper Function to Discretize State ---
def discretize_state(state, bins):
    indices = []
    for i in range(len(state)):
        index = np.digitize(state[i], bins[i])
        indices.append(index)
    return tuple(indices)

# --- Play using Trained Q-Table ---
episodes = 50
for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state, state_bins)
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_table[state])  # Always exploit best action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_state = discretize_state(next_state, state_bins)
        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
