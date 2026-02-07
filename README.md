# 🎓 RL Assignment: CartPole Mastery

Welcome to your Reinforcement Learning assignment! 

In this project, you will step beyond simply "solving" the CartPole environment. Your goal is to design a reward function that trains an agent to be **robust, smooth, and precise**.

![CartPole](https://gymnasium.farama.org/_images/cart_pole.gif)

---

## 📂 File Structure (What's what?)

*   `student_assignment_dqn.py`: **[🛠️ Your Workspace]** This is the **ONLY** file you need to modify. It contains the DQN training loop and the reward function you need to design.
*   `evaluate_performance.py`: **[📊 Score Card]** Runs your trained model to calculate a comprehensive score (Survival, Centering, Smoothness).
*   `evaluate_challenge.py`: **[🌪️ The Challenge]** Tests your agent in a "windy" environment where random forces push the pole.
*   `cartpole_basic.py`: **[📉 Baseline]** A simple rule-based agent (if-else logic) for comparison.
*   `requirements_*.txt`: Dependency files for Linux and Windows.

---

## 🚀 Step-by-Step Guide

### 1. Environment Setup

First, create a clean Python environment to run the project.

**Using Conda:**
```bash
# 1. Create environment
conda create -n rl_hw python=3.10 -y

# 2. Activate environment
conda activate rl_hw
```

**Option 1: Auto Install (Recommended - auto-detects GPU/CPU)**
```bash
# Linux/Mac one-liner: auto-detects GPU and installs the appropriate PyTorch version
bash install.sh
```

**Option 2: Manual Install**

*Option A: GPU environment (requires NVIDIA GPU + CUDA)*
```bash
# For Linux/Mac:
pip install -r requirements_linux.txt

# For Windows:
pip install -r requirements_win.txt
```

*Option B: CPU environment (no GPU, or CUDA install failed)*

If your machine does not have an NVIDIA GPU, or the GPU version fails to install, install the CPU-only version of PyTorch first:
```bash
# 1. Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining dependencies (torch is already installed, pip will skip it)
# For Linux/Mac:
pip install -r requirements_linux.txt
# For Windows:
pip install -r requirements_win.txt
```

> **Note**: The model in this project is very small (3-layer fully connected network). CPU training is perfectly sufficient — no GPU required to complete all tasks.

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, Device: {\"cuda\" if torch.cuda.is_available() else \"cpu\"}')"
```

---

### 2. Dry Run: Test the Baseline

Before training anything, let's see how a "dumb" rule-based agent performs. This gives you a baseline score to beat.

**Run the simple agent:**
```bash
python cartpole_basic.py
```
*Observation: It survives, but wobbles significantly and drifts off-center.*

**Check its score:**
```bash
python evaluate_performance.py --baseline
```
*Expected Score: Around 40-50 points. It fails the Centering and Smoothness criteria.*

---

### 3. First Training Run (No changes)

Now, let's run the DQN training script as-is to make sure everything works.

```bash
python student_assignment_dqn.py
```
*   **What happens**: It trains a DQN agent for 2000 episodes using a "Dummy Reward" (currently just basic survival).
*   **Output**: It saves a model file named `student_model.pth`.
*   **Plot**: It generates `student_training_curve.png` showing the reward over time.

---

### 4. 🧠 Your Task: Reward Engineering

This is the core of the assignment. 

1.  Open `student_assignment_dqn.py` in your editor.
2.  Locate the function `calculate_custom_reward`.
3.  **MODIFY IT!** 

Currently, the reward logic is very basic. You need to design a reward function that encourages:
*   **Centering**: Penalize the agent if `state[0]` (Cart Position) is far from 0.
*   **Stability**: Penalize the agent if `state[3]` (Pole Angular Velocity) is high.

**Example Logic (Pseudocode):**
```python
reward = 1.0  # Survival
reward -= abs(cart_position) * 0.5  # Penalize drift
reward -= abs(pole_velocity) * 0.1  # Penalize wobble
```

**After modifying the code, re-run the training:**
```bash
python student_assignment_dqn.py
```

---

### 5. Self-Evaluation

How good is your new model?

**1. Quantitative Score (The Grade):**
```bash
python evaluate_performance.py
```
*   **Target Score**: > 75
*   It evaluates 50 episodes and scores you on Survival, Centering, and Smoothness.

**2. Visual Inspection (The Eye Test):**
Want to see your agent in action? Add the `--render` flag:
```bash
python evaluate_performance.py --render
```
*   Watch: Does the cart stay in the middle? Does the pole shake or stay still?

**3. The Robustness Challenge (The Final Boss):**
Can your agent survive being pushed?
```bash
python evaluate_challenge.py
```
*   Random forces will push the pole. A robust policy should recover quickly.
*   **Pass Condition**: Survive > 300 steps on average.

---

## 🏆 Grading Criteria

| Component | Weight | Description |
|-----------|--------|-------------|
| **Survival** | 40% | Can it stay alive for 500 steps? |
| **Centering** | 30% | Does it stay near x=0? (Penalty if \|x\| > 0.2) |
| **Smoothness** | 30% | Is the movement stable? (Penalty if \|angular_vel\| > 0.2) |
| **Robustness** | Pass/Fail | Must survive > 300 steps in `evaluate_challenge.py` |

## 🙏 Acknowledgments & References

This assignment is adapted from the classic **CartPole-v1** environment provided by [Farama Foundation Gymnasium](https://gymnasium.farama.org/).

Special thanks to the open-source community for the initial implementations of DQN algorithms.
*   [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
*   [PyTorch Reinforcement Learning Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
