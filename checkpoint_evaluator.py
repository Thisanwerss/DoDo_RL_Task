# compare_models.py

from src.environment import create_env
from src.agent import QLearningAgent
from config import CONFIG

def evaluate_model(model_path, title):
    """
    Evaluate a specific model and show its performance.
    
    Args:
        model_path (str): Path to the saved model file
        title (str): Title to display during evaluation
    """
    print(f"\n{'='*50}")
    print(f"Evaluating {title}")
    print(f"{'='*50}")
    
    # Create the environment with rendering
    env = create_env(CONFIG["env_name"], render_mode="human")
    
    # Initialize the agent
    agent = QLearningAgent(env, CONFIG)
    
    # Load the specified model
    agent.load(model_path)
    
    # Evaluate the agent
    agent.evaluate(render=True, episodes=50)
    
    print(f"\nFinished evaluating {title}")
    input("Press Enter to continue to the next model...")

def main():
    print("\n🚀 Starting CartPole Model Comparison")
    
    # Define the models to compare
    models = [
        ("models/q_table_10percent.pkl", "Agent at 10% Training"),
        (CONFIG["model_path"], "Fully Trained Agent (100%)"),
    ]
    
    # Evaluate each model
    for model_path, title in models:
        evaluate_model(model_path, title)
    
    print("\n🎉 Comparison complete!")

if __name__ == "__main__":
    main()