import os
import sys
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

from langchain_core.messages import HumanMessage

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.graph import app

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set. Please create a .env file.")
        return

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is not set. Please add it to your .env file.")
        return

    print("AlphaSeeker MVP Agent")
    print("---------------------")
    query = input("Enter your request (e.g., 'Analyze AAPL for the last 1y'): ")
    
    if not query:
        print("No query provided. Exiting.")
        return
        
    print(f"\nProcessing: {query}...\n")
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "ticker": None,
        "period": None,
        "market_data_path": None,
        "chart_path": None,
        "financials_path": None,
        "peer_data_path": None,
        "company_profile_path": None,
        "plan": None,
        "source_metadata": {},
        "sections": {},
        "research_data": {},
        "research_brief": {},
        "report_content": None,
        "report_path": None,
        "error": None
    }
    
    try:
        final_state = app.invoke(initial_state)
        
        if final_state.get("error"):
            print(f"Error encountered: {final_state['error']}")
        else:
            print("Success!")
            print(f"Report saved to: {final_state.get('report_path')}")
            if final_state.get("chart_path"):
                print(f"Chart saved to: {final_state.get('chart_path')}")
            
            # Print executive summary
            report = final_state.get("report_content")
            if report and hasattr(report, "investment_summary"):
                print("\n--- EXECUTIVE SUMMARY ---")
                print(f"Ticker: {report.ticker}")
                print(f"Recommendation: {report.recommendation}")
                print(f"Target Price: {report.target_price}")
                print("\nInvestment Thesis:")
                print(report.mispricing_thesis)
                print("\nKey Catalysts:")
                print(report.key_catalysts)
            
    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
