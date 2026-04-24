from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

# Import our compiled LangGraph orchestrator and offline PDF tool
from main import app as orchestrator_app
from tools.pdf_parser import extract_text_from_pdf

# Initialize the FastAPI app
app = FastAPI(
    title="SentientChain API",
    description="Multimodal Multi-Agent Supply Chain Risk Orchestrator",
    version="1.0.0"
)

# --- Data Models (Input/Output Validation) ---
class AnalysisRequest(BaseModel):
    text_payload: Optional[str] = None
    pdf_filepath: Optional[str] = None

class AnalysisResponse(BaseModel):
    status: str
    final_report: str
    processed_text_preview: str

# --- Endpoints ---
@app.get("/health")
def health_check():
    """Simple endpoint to verify the API is running."""
    return {"status": "Operational", "service": "SentientChain Agent Router"}

@app.post("/analyze", response_model=AnalysisResponse)
def trigger_agent_workflow(request: AnalysisRequest):
    """
    Triggers the LangGraph multi-agent workflow. 
    Accepts either raw text or a path to a local PDF document.
    """
    # 1. Input Validation & Extraction
    if not request.text_payload and not request.pdf_filepath:
        raise HTTPException(status_code=400, detail="Must provide either text_payload or pdf_filepath")

    input_text = ""
    
    if request.pdf_filepath:
        if not os.path.exists(request.pdf_filepath):
            raise HTTPException(status_code=404, detail=f"PDF not found at {request.pdf_filepath}")
        print(f"API INGESTION: Parsing PDF at {request.pdf_filepath}")
        input_text = extract_text_from_pdf(request.pdf_filepath)
    else:
        input_text = request.text_payload

    # 2. Invoke the Orchestrator
    try:
        print("API: Routing payload to LangGraph...")
        initial_state = {"raw_news_event": input_text}
        
        # This synchronously runs the entire Watcher -> Quant -> RAG -> Synthesizer graph
        final_state = orchestrator_app.invoke(initial_state)
        
        report = final_state.get("final_report")
        
        # Handle the early termination case (Watcher found no risk)
        if not report:
             return AnalysisResponse(
                 status="Success - No Risk Detected", 
                 final_report="The Watcher agent determined this event does not pose a supply chain risk. Workflow terminated to save compute.",
                 processed_text_preview=input_text[:100] + "..."
             )

        # Handle the full success case
        return AnalysisResponse(
            status="Success - Risk Mitigated",
            final_report=report,
            processed_text_preview=input_text[:100] + "..."
        )
        
    except Exception as e:
        print(f"API ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Agent Orchestration Error")