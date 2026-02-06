import gymnasium
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm

# --- Neural Network Definition ---
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

def run_challenge(model_path="student_model.pth", num_episodes=20):
    """
    Runs the agent in an environment where random external forces (disturbances)
    are applied to the pole/cart occasionally.
    """
    print("="*50)
    print("      DISTURBANCE CHALLENGE STARTING...      ")
    print("="*50)

    # Use render_mode='human' to let students SEE the push, or None for speed
    # We will set it to None for grading speed, but 'human' is fun for demo.
    env = gymnasium.make('CartPole-v1') 
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DQN(state_dim, action_dim)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error: Could not load model. {e}")
        return

    survived_disturbances = 0
    total_disturbances = 0
    total_steps_survived = []

    for episode in tqdm(range(num_episodes), desc="Challenge Episodes"):
        state, _ = env.reset()
        done = False
        steps = 0
        
        # print(f"Episode {episode+1}/{num_episodes}...", end=" ")
        
        while not done:
            steps += 1
            
            # --- AGENT ACTION ---
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # --- APPLY DISTURBANCE ---
            # Random chance to push the pole
            if np.random.rand() < 0.02:  # 2% chance per step
                total_disturbances += 1
                
                # Apply a random "push" to the pole's angular velocity (state[3])
                # Magnitude: Random between -1.5 and 1.5 rad/s (pretty strong push!)
                push = np.random.uniform(-1.5, 1.5)
                
                # We need to access the underlying state to modify it
                # Note: gymnasium usually protects state, we use unwrapped or direct assignment if possible
                # In CartPole-v1, we can modify env.unwrapped.state
                
                current_internal_state = list(env.unwrapped.state)
                current_internal_state[3] += push  # Add to angular velocity
                env.unwrapped.state = np.array(current_internal_state)
                
                # print(f" [PUSH! {push:.2f}]", end="") 

            done = terminated or truncated
            state = next_state
        
        total_steps_survived.append(steps)
        # print(f"Survived {steps} steps.")

    env.close()
    
    avg_steps = np.mean(total_steps_survived)
    
    print("\n" + "="*50)
    print("           CHALLENGE RESULTS           ")
    print("="*50)
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Survival Steps (with disturbances): {avg_steps:.1f} / 500.0")
    print(f"Estimated Disturbance Survival Rate: {min(100, avg_steps/500*100):.1f}%")
    
    if avg_steps > 300:
        print("\n✅ PASSED: The agent is robust enough!")
    else:
        print("\n❌ FAILED: The agent falls too easily when pushed.")
        print("Hint: Try using Domain Randomization during training.")
    print("="*50)

if __name__ == "__main__":
    run_challenge()
