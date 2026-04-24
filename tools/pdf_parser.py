import os
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a local PDF file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find PDF at {file_path}")
        
    print(f"--- INGESTION: Reading offline PDF from {file_path} ---")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Combine all pages into a single string for the Watcher Agent
    full_text = "\n".join([page.page_content for page in pages])
    return full_text

# Create a dummy PDF generator for testing
def create_dummy_pdf():
    """Generates a sample PDF to test our ingestion pipeline."""
    from fpdf import FPDF # Requires: pip install fpdf2
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    
    text = """
    URGENT: PORT AUTHORITY MEMORANDUM
    Date: October 24, 2026
    Subject: Immediate Closure of Veracruz Port Terminals
    
    Due to an unpredicted Category 4 hurricane making landfall, all operations at the Veracruz shipping terminals have been suspended indefinitely. All incoming freight vessels carrying industrial metals, including silver and copper, are being redirected to alternative ports. Expect minimum delays of 7 to 10 days for all raw material supply chains reliant on this transit corridor.
    """
    pdf.multi_cell(0, 10, text)
    
    os.makedirs("data", exist_ok=True)
    file_path = "data/urgent_port_memo.pdf"
    pdf.output(file_path)
    return file_path