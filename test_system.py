from main import app
from tools.pdf_parser import create_dummy_pdf, extract_text_from_pdf
from core.evaluator import evaluate_report

def run_evaluation_pipeline():
    print("Initializing Automated AI Evaluation Pipeline...\n")
    
    # 1. Setup the test data
    pdf_path = create_dummy_pdf()
    source_text = extract_text_from_pdf(pdf_path)
    initial_state = {"raw_news_event": source_text}
    
    # 2. Run the LangGraph system
    final_state = app.invoke(initial_state)
    report = final_state.get("final_report")
    history = final_state.get("historical_context", [])
    
    if not report:
        print("Test Failed: System did not generate a report.")
        return
        
    # 3. Run the LLM-as-a-Judge Evaluation
    eval_results = evaluate_report(source_text, report, history)
    # Bulletproof extraction: force integers, default to 0 if the model messes up
    try:
        f_score = int(eval_results.get('faithfulness_score', 0))
        r_score = int(eval_results.get('relevance_score', 0))
    except (TypeError, ValueError):
        f_score, r_score = 0, 0
        print("WARNING: Model failed to output valid integers.")
        
    # 4. Assertions (The CI/CD part)
    print("\n" + "="*40)
    print("EVALUATION RESULTS:")
    print(f"Faithfulness Score: {f_score}/10")
    print(f"Relevance Score: {r_score}/10")
    print(f"Judge Reasoning: {eval_results.get('reasoning', 'No reasoning provided.')}")
    print("="*40)
    
    # Simulate a CI/CD pipeline failure if the model hallucinates
    assert f_score >= 8, f"CRITICAL FAILURE: Hallucination detected. Score: {f_score}"
    assert r_score >= 8, f"CRITICAL FAILURE: Irrelevant mitigations. Score: {r_score}"
    
    print("\n✅ All AI automated tests passed! System is ready for production deployment.")

if __name__ == "__main__":
    run_evaluation_pipeline()