import os
import sys
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

from langchain_core.messages import HumanMessage

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.supervisor.graph import app

def main() -> None:
    """Entry point for AlphaSeeker — routes all queries through the Supervisor agent."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set. Please create a .env file.")
        return

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is not set. Please add it to your .env file.")
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
        "intent": None,
        "sub_agents_needed": None,
        "classified_entities": None,
        "agent_results": None,
        "final_response": None,
        "error": None,
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
