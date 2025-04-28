import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make('CartPole-v1')

episodes = 5
rewards_per_episode = []

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        env.render()

        pole_angle = state[2]

        if pole_angle < 0:
            action = 0
        else:
            action = 1

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step += 1

        state = next_state

    rewards_per_episode.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {step}")

env.close()

# Plot
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Simple Rule-Based Agent Performance')
plt.show()
