import os
import sys
import argparse
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

# Ensure src is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.harness import HarnessRequest, run_harness
from src.shared.model_config import get_missing_provider_env_vars


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AlphaSeeker research pipelines.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="The research question or task to execute.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for AlphaSeeker — routes all queries through the Harness runtime."""
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

    query = args.prompt
    if not query:
        query = input("Enter your request (e.g., 'Analyze AAPL', 'US macro outlook', 'Crude oil analysis'): ")

    if not query:
        print("No query provided. Exiting.")
        return

    print(f"\nProcessing: {query}...\n")

    try:
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

    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
