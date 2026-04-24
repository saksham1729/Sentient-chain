from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # NEW: Required for HITL
from agents.watcher import analyze_supply_chain_event
from agents.quant import analyze_quantitative_impact
from agents.synthesizer import generate_mitigation_report
from core.memory import retrieve_context
from tools.pdf_parser import extract_text_from_pdf, create_dummy_pdf
import os
from agents.telemetry import check_maritime_telemetry # NEW IMPORT
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Define the Graph State
class SupplyChainState(TypedDict):
    asset_ticker: str
    raw_news_event: str
    risk_assessment: Optional[dict]
    quantitative_impact: Optional[str]
    historical_context: Optional[List[str]] #Added for RAG data retrieval
    final_report: Optional[str]   #Added for the synthesizer output
    telemetry_data: Optional[str] # NEW: For storing maritime telemetry insights

# 2. Define the Nodes (The actual work being done)
def watcher_node(state: SupplyChainState):
    print(f"--- WATCHER NODE: Analyzing Text for {state['asset_ticker']} ---")
    assessment = analyze_supply_chain_event(state["raw_news_event"], state["asset_ticker"])
    
    # Apply our fix for the Llama 3.2 nested properties quirk
    if "properties" in assessment:
        assessment = assessment["properties"]
        
    return {"risk_assessment": assessment}

def quant_node(state: SupplyChainState):
    print(f"--- QUANT NODE: Fetching Market Data & Analyzing Impact for {state['asset_ticker']} ---")
    impact = analyze_quantitative_impact(state["risk_assessment"], state["asset_ticker"])
    return {"quantitative_impact": impact}

#Add the Retriever Node
# def retriever_node(state: SupplyChainState):
#     print(f"--- RETRIEVER NODE: Fetching Historical Context for {state['asset_ticker']} ---")
#     # Search the vector DB using the Watcher's summary
#     search_query = state["risk_assessment"].get("summary", state["raw_news_event"])
#     historical_data = retrieve_context(search_query, k=1)
#     return {"historical_context": historical_data}
def retriever_node(state: SupplyChainState):
    print(f"--- RETRIEVER NODE: Fetching Context for {state['asset_ticker']} ---")
    search_query = state["risk_assessment"].get("summary", state["raw_news_event"])
    
    # 1. Try local Vector DB (History)
    history = retrieve_context(search_query, k=1)
    
    # 2. Perform Live Web Search for breaking context
    print(f"--- WEB SEARCH NODE: Gathering live intelligence on {state['asset_ticker']} ---")
    try:
        search = DuckDuckGoSearchRun()
        # We combine the ticker and the event to get highly targeted web results
        web_query = f"{state['asset_ticker']} {search_query}"
        web_context = search.invoke(web_query)
        
        # Inject the live web data as a highly prioritized context block
        live_intel = f"LIVE WEB SEARCH INTELLIGENCE: {web_context}"
        history.append(live_intel)
    except Exception as e:
        print(f"Web search failed: {e}")
        history.append("LIVE WEB SEARCH INTELLIGENCE: Unavailable.")
        
    return {"historical_context": history}


#Telemetry Node
def telemetry_node(state: SupplyChainState):
    summary = state["risk_assessment"].get("summary", "")
    telemetry = check_maritime_telemetry(summary)
    return {"telemetry_data": telemetry}

#Add the Synthesizer Node
def synthesizer_node(state: SupplyChainState):
    print("--- SYNTHESIZER NODE: Drafting Executive Report ---")
    # We pass the new telemetry data into the prompt context dynamically
    state["historical_context"].append(f"LIVE TELEMETRY: {state['telemetry_data']}")
    report = generate_mitigation_report(state)
    return {"final_report": report}

def route_risk(state: SupplyChainState):
    if state.get("risk_assessment", {}).get("is_risk"):
        print(">>> ROUTER: High risk. Triggering Advanced Pipeline.")
        return "telemetry_node" # Route to Telemetry first
    return END

# 4. Assemble the Graph
workflow = StateGraph(SupplyChainState)

workflow.add_node("watcher_node", watcher_node)
workflow.add_node("telemetry_node", telemetry_node)
workflow.add_node("quant_node", quant_node)
workflow.add_node("retriever_node", retriever_node) # REGISTER NODE
workflow.add_node("synthesizer_node", synthesizer_node)
# Set the entry point
workflow.add_edge(START, "watcher_node")

# Add the conditional routing logic
workflow.add_conditional_edges(
    "watcher_node", 
    route_risk, 
    {
        "telemetry_node": "telemetry_node",
        END: END
    }
)

# Route Quant to Synthesizer, then Synthesizer to END
workflow.add_edge("telemetry_node", "quant_node")
workflow.add_edge("quant_node", "retriever_node")
workflow.add_edge("retriever_node", "synthesizer_node")
workflow.add_edge("synthesizer_node", END)

# Compile the engine


# # --- Run the multi-agent system ---
# if __name__ == "__main__":
#     # Test Scenario 1: A massive disruption
#     sample_news = "Workers at the Penasquito silver mine in Mexico have initiated a sudden labor strike over profit-sharing disputes, halting all extraction operations immediately."
    
#     # Uncomment this to test the "No Risk" routing path:
#     # sample_news = "The local community held a peaceful festival near the silver mine, operations continued as normal."
    
#     initial_state = {"raw_news_event": sample_news}
    
#     print("\nStarting SentientChain Execution...\n" + "="*40)
#     final_state = app.invoke(initial_state)
    
#     print("\n" + "="*40 + "\nFINAL SYSTEM STATE:")
#     if "final_report" in final_state:
#         print(final_state["final_report"])
#     else:
#         print("No risk detected, workflow terminated early.")

# if __name__ == "__main__":
#     # 1. Generate our dummy unstructured PDF
#     pdf_path = create_dummy_pdf()
    
#     # 2. Extract the text completely offline
#     extracted_event_text = extract_text_from_pdf(pdf_path)
    
#     # 3. Feed the parsed text into our LangGraph orchestrator
#     initial_state = {"raw_news_event": extracted_event_text}
    
#     print("\nStarting SentientChain Execution...\n" + "="*40)
#     final_state = app.invoke(initial_state)
    
#     print("\n" + "="*40 + "\nFINAL SYSTEM REPORT:\n")
#     if "final_report" in final_state:
#         print(final_state["final_report"])
#     else:
#         print("No risk detected, workflow terminated early.")

# NEW: Setup Memory and compile with an interruption
memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["synthesizer_node"] # Pause right before synthesis
)
if __name__ == "__main__":
    print("\n" + "="*50)
    print("WELCOME TO SENTIENTCHAIN ORCHESTRATOR")
    print("="*50)
    
    # NEW: Interactive User Prompts
    user_ticker = input("1. Enter the Stock/Asset Ticker (e.g., PARAS.NS, AAPL, TSLA): ").strip().upper()
    if not user_ticker.endswith(".NS") and not user_ticker.endswith(".BO") and not user_ticker in ["AAPL", "TSLA"]: # Add your US exceptions here
        print(f"Auto-formatting {user_ticker} to {user_ticker}.NS for Yahoo Finance...")
        user_ticker += ".NS"
    user_event = input(f"2. Enter the breaking news event impacting {user_ticker}: ").strip()
    
    config = {"configurable": {"thread_id": "dynamic_incident_001"}}
    
    print("\n[STEP 1] Starting Autonomous Execution...\n" + "="*50)
    initial_state = {
        "asset_ticker": user_ticker, 
        "raw_news_event": user_event
    }
    
    events = app.invoke(initial_state, config=config)
    
    print("\n" + "="*50)
    print("🛑 SYSTEM PAUSED: HUMAN-IN-THE-LOOP CHECKPOINT REACHED")
    print(f"The AI has gathered Telemetry, Quant Data for {user_ticker}, and History.")
    
    input("\nPress ENTER to approve and generate the final Executive Report...")
    
    print("\n[STEP 2] Resuming workflow...\n")
    final_state = app.invoke(None, config=config)
    
    print("\nFINAL SYSTEM REPORT:\n")
    print(final_state.get("final_report", "Report failed to generate."))