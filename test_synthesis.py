import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.supervisor.synthesizer import SynthesisInput, run_synthesis

def test_synthesizer():
    print("--- Testing Supervisor Synthesizer ---")
    
    # 1. Single agent passthrough test
    print("\n[Test 1] Single Agent Passthrough")
    single_input = SynthesisInput(
        user_prompt="Analyze AAPL's recent earnings",
        agent_results={
            "equity": "## AAPL Earnings Report\n\nRevenue grew 5% YoY. Margins improved."
        },
        primary_intent="equity"
    )
    res1 = run_synthesis(single_input)
    print(f"Mode: {res1.mode}")
    print(f"Output:\n{res1.final_response[:100]}...\n")
    
    # 2. Multi-agent synthesis test
    print("\n[Test 2] Multi-Agent Synthesis")
    multi_input = SynthesisInput(
        user_prompt="How will interest rate cuts affect the gold market?",
        agent_results={
            "macro": "## Macro Brief\n\nThe Fed is expected to cut rates by 50bps this year due to cooling inflation.",
            "commodity": "## Gold Brief\n\nGold traditionally performs well in a falling rate environment as the opportunity cost of holding non-yielding assets decreases."
        },
        primary_intent="commodity"
    )
    res2 = run_synthesis(multi_input)
    print(f"Mode: {res2.mode}")
    print(f"Output:\n{res2.final_response}\n")

if __name__ == "__main__":
    test_synthesizer()
