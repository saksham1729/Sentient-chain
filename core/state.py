from typing import TypedDict, List, Optional
from agents.watcher import RiskAssessment

class SupplyChainState(TypedDict):
    """
    The shared state that gets passed between all our agents.
    """
    raw_news_event: str
    risk_assessment: Optional[RiskAssessment]
    historical_context: List[str]  # The RAG data will go here later
    quantitative_impact: Optional[str] # The Quant agent's analysis
    final_report: Optional[str]