import os
import sys
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.supervisor.graph import app
from src.shared.model_config import get_required_provider_env_vars

def main() -> None:
    """Entry point for AlphaSeeker — routes all queries through the Supervisor agent."""
    required_env_vars = get_required_provider_env_vars()
    missing_env_vars = [env for env in sorted(required_env_vars) if not os.getenv(env)]
    if missing_env_vars:
        print("Error: Missing required API key env vars for configured model providers:")
        for env in missing_env_vars:
            print(f"  - {env}")
        print("Create a .env from .env.example and fill the required keys.")
        return

    print("AlphaSeeker Multi-Agent System")
    print("------------------------------")
    query = input("Enter your request (e.g., 'Analyze AAPL', 'US macro outlook', 'Crude oil analysis'): ")
    
    if not query:
        print("No query provided. Exiting.")
        return
        
    print(f"\nProcessing: {query}...\n")
    
    # The Supervisor handles everything: classification → sub-agent dispatch → synthesis
    initial_state = {
        "user_prompt": query,
    }
    
    try:
        final_state = app.invoke(initial_state)
        
        if final_state.get("error"):
            print(f"Error encountered: {final_state['error']}")
        else:
            print("Success!")
            
            # Print the synthesized response
            final_response = final_state.get("final_response")
            if final_response:
                print("\n--- RESPONSE ---")
                print(final_response)
            
            # List any sub-agent reports generated
            agent_results = final_state.get("agent_results", {})
            if agent_results:
                print(f"\nSub-agents that ran: {', '.join(agent_results.keys())}")
            
    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
