from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class RiskAssessment(BaseModel):
    is_risk: bool = Field(description="True if the event disrupts the supply chain.")
    risk_type: str = Field(description="Category: e.g., Geopolitical, Labor, Natural Disaster, None.")
    severity: int = Field(description="Impact severity from 1 to 10.")
    summary: str = Field(description="A one-sentence summary of the impact.")

def analyze_supply_chain_event(event_text: str, ticker: str):
    # Fix 1: Use OllamaLLM instead of the deprecated Ollama class
    # Fix 2: Set format="json" and temperature=0 to force strict, repeatable JSON outputs
    llm = OllamaLLM(model="llama3.2", format="json", temperature=0)
    
    parser = JsonOutputParser(pydantic_object=RiskAssessment)
    
    # FIX: Removed "silver market" and added {ticker}
    prompt = PromptTemplate(
        template="""You are an analytical Risk Agent monitoring the market for the asset: {ticker}.
        Analyze the event and output a JSON object containing your assessment of how it impacts this specific asset's supply chain or value.
        
        Event: {event}
        
        {format_instructions}
        
        Respond ONLY with valid JSON. Do not include introductory text or markdown formatting.
        """,
        input_variables=["event", "ticker"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    result = chain.invoke({"event": event_text, "ticker": ticker})
    # Handle Llama 3.2 nesting quirk
    if "properties" in result:
        result = result["properties"]

    return result

if __name__ == "__main__":
    sample_news = "Workers at the Penasquito silver mine in Mexico have initiated a sudden labor strike over profit-sharing disputes, halting all extraction operations immediately."
    
    print("Analyzing event...")
    # FIX: Added a dummy ticker argument to the test function call
    assessment = analyze_supply_chain_event(sample_news, "SILVERBEES.NS")
    print("\nWatcher Agent Output:")
    print(assessment)