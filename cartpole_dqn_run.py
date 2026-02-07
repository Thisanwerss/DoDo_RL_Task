import gymnasium
import torch
import torch.nn as nn
import numpy as np
import time
import argparse

# --- Neural Network (same as before) ---
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

# --- Rule-based policy (simple baseline) ---
def rule_based_policy(state):
    """Push left if pole leans left, push right if pole leans right."""
    pole_angle = state[2]
    return 0 if pole_angle < 0 else 1

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Run CartPole agent with visualization")
parser.add_argument("--baseline", action="store_true",
                    help="Use rule-based baseline policy instead of the trained DQN model")
parser.add_argument("--model", type=str, default="student_model.pth",
                    help="Path to the trained model file (default: student_model.pth)")
parser.add_argument("--episodes", type=int, default=10,
                    help="Number of episodes to run (default: 10)")
args = parser.parse_args()

# --- Load the environment ---
env = gymnasium.make('CartPole-v1', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# --- Load model or use baseline ---
model = None
device = None

if args.baseline:
    print("Running with rule-based baseline policy")
else:
    # Detect device: use GPU if available, otherwise fall back to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained model
    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Evaluation mode
    print(f"Loaded model from: {args.model}")

# --- Play ---
max_steps = 500  # Cap the number of steps per episode

for episode in range(args.episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    # For tracking state variables
    positions = []
    velocities = []
    angles = []
    angular_velocities = []
    
    while not done and step < max_steps:
        step += 1
        # Track state variables
        positions.append(state[0])
        velocities.append(state[1])
        angles.append(state[2])
        angular_velocities.append(state[3])
        
        # Select action
        if args.baseline:
            action = rule_based_policy(state)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Print state information every 50 steps
        if step % 50 == 0:
            print(f"Step {step}:")
            print(f"  Cart Position: {next_state[0]:.4f}")
            print(f"  Cart Velocity: {next_state[1]:.4f}")
            print(f"  Pole Angle: {next_state[2]:.4f}")
            print(f"  Pole Angular Velocity: {next_state[3]:.4f}")
            
        done = terminated or truncated
        state = next_state
        total_reward += reward
        
        # Add a small delay to see the movement clearly
        time.sleep(0.01)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {step}")
    
    # Print episode statistics
    if len(positions) > 0:
        print(f"  Position range: [{min(positions):.2f}, {max(positions):.2f}]")
        print(f"  Average abs position: {np.mean(np.abs(positions)):.2f}")

env.close()