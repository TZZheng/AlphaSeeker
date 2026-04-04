import os
import sys
import argparse
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.supervisor.graph import app
from src.shared.model_config import get_missing_provider_env_vars
from src.harness import HarnessRequest, run_harness

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AlphaSeeker research pipelines.")
    parser.add_argument(
        "--runtime",
        choices=("legacy", "harness"),
        default="legacy",
        help="Which runtime to execute.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for AlphaSeeker — routes all queries through the Supervisor agent."""
    args = _parse_args(argv)
    missing_requirements = get_missing_provider_env_vars()
    if missing_requirements:
        print("Error: Missing required API key env vars for configured model providers:")
        for provider_label, requirement in missing_requirements.items():
            print(f"  - {provider_label}: {requirement}")
        print("Create a .env from .env.example and fill the required keys.")
        return

    print("AlphaSeeker Multi-Agent System")
    print("------------------------------")
    print(f"Runtime: {args.runtime}")
    query = input("Enter your request (e.g., 'Analyze AAPL', 'US macro outlook', 'Crude oil analysis'): ")
    
    if not query:
        print("No query provided. Exiting.")
        return
        
    print(f"\nProcessing: {query}...\n")
    
    try:
        if args.runtime == "harness":
            response = run_harness(HarnessRequest(user_prompt=query))
            if response.status == "failed":
                print(f"Error encountered: {response.error or response.stop_reason or 'Harness failed.'}")
            else:
                print("Success!")
                if response.final_report_path and os.path.exists(response.final_report_path):
                    print("\n--- RESPONSE ---")
                    with open(response.final_report_path, "r", encoding="utf-8") as fh:
                        print(fh.read())
                if response.run_root:
                    print(f"\nRun root: {response.run_root}")
                if response.root_agent_path:
                    print(f"Root agent workspace: {response.root_agent_path}")
                if response.final_report_path:
                    print(f"Final report saved: {response.final_report_path}")
        else:
            # The Supervisor handles everything: classification → sub-agent dispatch → synthesis
            initial_state = {
                "user_prompt": query,
            }
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
