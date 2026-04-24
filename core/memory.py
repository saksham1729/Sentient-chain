import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Define the local path for our Vector DB
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

def get_vector_store():
    """Initializes and returns the ChromaDB connection using local Ollama embeddings."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # We use Langchain's Chroma integration to manage the underlying database
    vector_store = Chroma(
        collection_name="supply_chain_history",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    return vector_store

def seed_database():
    """Injects sample historical data into the vector database."""
    vector_store = get_vector_store()
    
    # In a real app, you would load these from PDFs using Langchain Document Loaders
    sample_documents = [
        Document(
            page_content="In 2023, a two-week strike at a major Peruvian silver mine caused a 4% spike in global silver spot prices. Mitigation involved rerouting supply from secondary stockpiles in Nevada.",
            metadata={"source": "historical_report_2023", "event_type": "labor_strike", "commodity": "silver"}
        ),
        Document(
            page_content="The 2021 Suez Canal blockage delayed silver freight shipments by 14 days. Companies utilizing air-freight contingencies bypassed the delay but incurred a 300% premium on shipping costs.",
            metadata={"source": "logistics_postmortem_2021", "event_type": "logistics_failure", "commodity": "general_freight"}
        )
    ]
    
    print("Embedding and storing historical documents...")
    vector_store.add_documents(sample_documents)
    print(f"Database seeded successfully at {DB_DIR}")

def retrieve_context(query: str, k: int = 1):
    """Searches the vector database for relevant historical context."""
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

if __name__ == "__main__":
    # Run this once to populate the local database
    seed_database()
    
    # Test the retrieval
    print("\nTesting Retrieval...")
    context = retrieve_context("What happened during past mining strikes?", k=1)
    print(f"Retrieved Context: {context[0]}")