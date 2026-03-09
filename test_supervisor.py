import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.supervisor.graph import classify_intent, SupervisorState

def test_routing():
    print("--- Testing Supervisor Router ---")
    
    prompts = [
        "Analyze AAPL's recent earnings",
        "How will interest rate cuts affect the gold market?",
        "Give me an overview of the macro economy"
    ]
    
    for prompt in prompts:
        state: SupervisorState = {"user_prompt": prompt}
        print(f"\nUser Prompt: '{prompt}'")
        try:
            result = classify_intent(state)
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Primary Intent: {result['intent']}")
                print(f"Sub-agents needed: {result['sub_agents_needed']}")
                for k, v in result['classified_entities'].items():
                    print(f"  {k} -> {v}")
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    test_routing()
