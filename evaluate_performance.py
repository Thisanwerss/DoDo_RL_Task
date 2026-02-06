import gymnasium
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm

# --- Neural Network Definition (Must match training) ---
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

def rule_based_policy(state):
    # Simple logic: Push left if leaning left, push right if leaning right
    pole_angle = state[2]
    return 0 if pole_angle < 0 else 1

def evaluate(model_path="student_model.pth", num_episodes=50, render=False, use_baseline=False):
    if not use_baseline and not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train your model first.")
        return

    render_mode = "human" if render else None
    env = gymnasium.make('CartPole-v1', render_mode=render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = None
    if not use_baseline:
        model = DQN(state_dim, action_dim)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading model structure: {e}")
            return
        print(f"Evaluating Model: {model_path}")
    else:
        print(f"Evaluating Baseline: Rule-Based Agent")

    print(f"Episodes: {num_episodes}")
    print("-" * 30)

    stats = {
        "rewards": [],
        "steps": [],
        "avg_abs_position": [],  # Metric for Centering
        "avg_abs_angle_vel": []  # Metric for Smoothness
    }

    for i in tqdm(range(num_episodes), desc="Evaluating Episodes"):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Episode metrics
        positions = []
        angle_vels = []

        while not done:
            if use_baseline:
                action = rule_based_policy(state)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Record state data
            positions.append(next_state[0])
            angle_vels.append(next_state[3])

            state = next_state
            episode_reward += reward
            steps += 1

        # Store episode stats
        stats["rewards"].append(episode_reward)
        stats["steps"].append(steps)
        stats["avg_abs_position"].append(np.mean(np.abs(positions)) if positions else 0)
        stats["avg_abs_angle_vel"].append(np.mean(np.abs(angle_vels)) if angle_vels else 0)
        
    env.close()

    # --- Calculate Final Scores ---
    avg_reward = np.mean(stats["rewards"])
    success_rate = sum(r >= 475 for r in stats["rewards"]) / num_episodes * 100 
    avg_centering_error = np.mean(stats["avg_abs_position"])
    avg_smoothness_error = np.mean(stats["avg_abs_angle_vel"])

    # Scoring Logic 
    # 1. Survival Score (0-40 pts): It's easy to survive, so we give less points here.
    score_survival = min(40, avg_reward / 500 * 40)
    
    # 2. Centering Score (0-30 pts): 
    # Target: average position deviation < 0.2 (It was 0.5 before)
    # If avg_centering_error > 0.2, score is 0.
    score_centering = 30 * max(0, 1 - (avg_centering_error / 0.2))
    
    # 3. Smoothness Score (0-30 pts): 
    # Target: average angular velocity < 0.2 rad/s (It was 1.0 before)
    # If avg_smoothness_error > 0.2, score is 0.
    score_smoothness = 30 * max(0, 1 - (avg_smoothness_error / 0.2))

    total_score = score_survival + score_centering + score_smoothness

    print("\n" + "="*40)
    print("       PERFORMANCE REPORT      ")
    print("="*40)
    print(f"Metrics (Average over {num_episodes} episodes):")
    print(f"  - Reward (Survival):  {avg_reward:.2f} / 500.0")
    print(f"  - Success Rate:       {success_rate:.1f}%")
    print(f"  - Centering Error:    {avg_centering_error:.4f} (Target < 0.2)")
    print(f"  - Smoothness Error:   {avg_smoothness_error:.4f} (Target < 0.2)")
    print("-" * 40)
    print("Scoring Breakdown:")
    print(f"  [+] Survival Score:   {score_survival:.1f} / 40")
    print(f"  [+] Centering Score:  {score_centering:.1f} / 30")
    print(f"  [+] Smoothness Score: {score_smoothness:.1f} / 30")
    print("="*40)
    print(f"  FINAL SCORE: {total_score:.1f} / 100")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Evaluate the rule-based baseline instead of the trained model")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()
    
    evaluate(render=args.render, use_baseline=args.baseline)
