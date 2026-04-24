from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def generate_mitigation_report(state: dict):
    llm = OllamaLLM(model="llama3.2", temperature=0.3)
    
    prompt = PromptTemplate(
        template="""You are the Lead AI Strategist.
        Synthesize the provided data into a concise Executive Report for the asset: {ticker}.
        
        --- INPUT DATA ---
        Original Event: {event}
        Risk Profile: {assessment}
        Quantitative Impact Thesis: {quant_impact}
        Retrieved Intelligence (Local & Web): {history}
        ------------------
        
        Format your response in Markdown using exactly these three headers:
        ### Executive Summary
        (Summarize the event and risk based strictly on the event and retrieved intelligence)
        
        ### Market Impact
        (Incorporate the quantitative impact thesis)
        
        ### Recommended Actions
        (Provide 2 immediate mitigations for {ticker}. 
        CRITICAL INSTRUCTION: Base your recommendations ON THE LIVE WEB SEARCH INTELLIGENCE. Absolutely do NOT mention silver, mining, or Nevada unless the LIVE WEB SEARCH explicitly mentions them in relation to {ticker}.)
        """,
        input_variables=["event", "assessment", "quant_impact", "history", "ticker"],
    )
    
    chain = prompt | llm
    
    # Format the list of historical strings into a single text block
    history_data = state.get("historical_context")
    history_text = "\n".join(history_data) if history_data else "No historical context available."
    
    result = chain.invoke({
        "event": state.get("raw_news_event"),
        "assessment": state.get("risk_assessment"),
        "quant_impact": state.get("quantitative_impact"),
        "history": history_text,
        "ticker": state.get("asset_ticker")
    })
    
    return result