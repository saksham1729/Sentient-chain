from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class EvaluationScore(BaseModel):
    faithfulness_score: int = Field(description="Score from 1-10 on whether the report hallucinated facts not in the source text.")
    relevance_score: int = Field(description="Score from 1-10 on whether the report provides actionable supply chain mitigations.")
    reasoning: str = Field(description="A concise 1-sentence explanation of the scores.")

def evaluate_report(source_text: str, report: str, history: list) -> dict:
    """Evaluates the final generated report against the original ground truth data."""
    print("\n--- EVALUATION HARNESS: Grading Final Output ---")
    
    # We use temperature=0 because evaluation must be strictly deterministic
    llm = OllamaLLM(model="llama3.2", format="json", temperature=0)
    parser = JsonOutputParser(pydantic_object=EvaluationScore)
    
   # --- Update inside core/evaluator.py ---
    prompt = PromptTemplate(
        template="""You are an impartial AI Evaluation Judge. 
        Evaluate the AI-generated report against the source data.
        
        [SOURCE TEXT]: {source}
        [RETRIEVED HISTORY]: {history}
        
        [AI GENERATED REPORT]: {report}
        
        Evaluate the report based on:
        1. Faithfulness: Did the report invent facts not present in the Source? (1-10)
        2. Relevance: Are the recommended actions highly relevant? (1-10)
        
        CRITICAL INSTRUCTION: Do NOT output the schema definitions. You MUST output actual integer numbers for the scores.
        
        Example Valid Output:
        {{
            "faithfulness_score": 9,
            "relevance_score": 8,
            "reasoning": "The report accurately reflected the source text."
        }}
        
        {format_instructions}
        """,
        input_variables=["source", "history", "report"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    
    history_str = "\n".join(history) if history else "None"
    result = chain.invoke({
        "source": source_text,
        "history": history_str,
        "report": report
    })
    
    if "properties" in result:
        result = result["properties"]
        
    return result