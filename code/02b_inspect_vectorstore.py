"""
Step 2b: Inspect Vector Store (Demonstration)
=============================================

This script demonstrates what's stored in the FAISS vector store.
It shows:
- Total number of document chunks
- Sample chunks with metadata
- Embedding dimensions
- Example similarity searches
- Source file breakdown

Run this after Step 2 to see what data is stored in the vector database.
"""

import os
from pathlib import Path
from collections import Counter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import sys

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_config import get_api_config, get_embedding_model

# Load environment variables
load_dotenv()


def load_vectorstore():
    """Load the vector store from disk."""
    vectorstore_path = Path("vectorstore")
    
    if not vectorstore_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {vectorstore_path}. "
            "Please run code/02_create_vectorstore.py first."
        )
    
    config = get_api_config()
    if not config:
        raise ValueError("API key not found. Set OPENAI_API_KEY or OPENROUTER_API_KEY")
    
    model_name = get_embedding_model(config["provider"])
    
    embedding_kwargs = {
        "model": model_name,
        "openai_api_key": config["api_key"]
    }
    
    if config["base_url"]:
        embedding_kwargs["openai_api_base"] = config["base_url"]
    
    embeddings = OpenAIEmbeddings(**embedding_kwargs)
    
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore, embeddings, config


def get_vectorstore_stats(vectorstore):
    """Get statistics about the vector store."""
    # Get all documents from the vector store
    # Note: FAISS doesn't have a direct method to get all docs
    # We'll use a broad search to get approximate count
    try:
        # Try to get document store
        if hasattr(vectorstore, 'docstore'):
            docstore = vectorstore.docstore
            if hasattr(docstore, '_dict'):
                total_docs = len(docstore._dict)
            else:
                total_docs = "Unknown (check FAISS index)"
        else:
            total_docs = "Unknown"
    except:
        total_docs = "Unknown"
    
    return total_docs


def show_sample_chunks(vectorstore, num_samples=5):
    """Show sample chunks from the vector store."""
    print("\n" + "=" * 80)
    print("SAMPLE CHUNKS FROM VECTOR STORE")
    print("=" * 80)
    
    # Use a generic query to retrieve some chunks
    sample_query = "customer support"
    results = vectorstore.similarity_search(sample_query, k=num_samples)
    
    print(f"\nRetrieved {len(results)} sample chunks (using query: '{sample_query}'):")
    print()
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        source_name = Path(source).name if source != 'Unknown' else 'Unknown'
        
        # Get chunk info
        chunk_num = doc.metadata.get('chunk', 'N/A')
        
        # Preview content
        content_preview = doc.page_content[:200].replace('\n', ' ')
        if len(doc.page_content) > 200:
            content_preview += "..."
        
        print(f"Chunk {i}:")
        print(f"  Source: {source_name}")
        print(f"  Full Path: {source}")
        if chunk_num != 'N/A':
            print(f"  Chunk Number: {chunk_num}")
        print(f"  Content Preview: {content_preview}")
        print(f"  Content Length: {len(doc.page_content)} characters")
        print()


def show_source_breakdown(vectorstore):
    """Show breakdown by source file."""
    print("\n" + "=" * 80)
    print("SOURCE FILE BREAKDOWN")
    print("=" * 80)
    
    # Get chunks from multiple queries to get a good sample
    queries = [
        "password reset",
        "pricing plans",
        "data privacy",
        "troubleshooting",
        "user guide"
    ]
    
    all_sources = []
    for query in queries:
        results = vectorstore.similarity_search(query, k=10)
        for doc in results:
            source = doc.metadata.get('source', 'Unknown')
            all_sources.append(Path(source).name if source != 'Unknown' else 'Unknown')
    
    # Count occurrences
    source_counts = Counter(all_sources)
    
    print("\nDocuments found in vector store (sample):")
    for source, count in source_counts.most_common():
        print(f"  {source}: {count} chunks (approximate)")


def demonstrate_similarity_search(vectorstore, embeddings):
    """Demonstrate similarity search with examples."""
    print("\n" + "=" * 80)
    print("SIMILARITY SEARCH DEMONSTRATION")
    print("=" * 80)
    
    test_queries = [
        "How do I reset my password?",
        "What are the pricing tiers?",
        "How do I cancel my subscription?",
        "What is your refund policy?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)
        
        # Perform similarity search
        results = vectorstore.similarity_search(query, k=3)
        
        print(f"Found {len(results)} relevant chunks:")
        
        for i, doc in enumerate(results, 1):
            source = Path(doc.metadata.get('source', 'Unknown')).name
            content_preview = doc.page_content[:150].replace('\n', ' ')
            if len(doc.page_content) > 150:
                content_preview += "..."
            
            print(f"\n  {i}. Source: {source}")
            print(f"     Preview: {content_preview}")


def show_embedding_info(embeddings, config):
    """Show information about embeddings."""
    print("\n" + "=" * 80)
    print("EMBEDDING INFORMATION")
    print("=" * 80)
    
    model_name = get_embedding_model(config["provider"])
    
    print(f"\nEmbedding Model: {model_name}")
    print(f"Provider: {config['provider'].upper()}")
    
    # Test embedding to get dimensions
    test_text = "This is a test sentence."
    try:
        test_embedding = embeddings.embed_query(test_text)
        dimension = len(test_embedding)
        print(f"Embedding Dimension: {dimension}")
        print(f"  (Each chunk is converted to a vector of {dimension} numbers)")
        print(f"\nExample embedding (first 10 values):")
        print(f"  {test_embedding[:10]}")
        print(f"  ... (showing first 10 of {dimension} values)")
    except Exception as e:
        print(f"Could not get embedding dimensions: {e}")


def main():
    """Main function to inspect vector store."""
    print("=" * 80)
    print("INSPECTING FAISS VECTOR STORE")
    print("=" * 80)
    print("\nThis script demonstrates what's stored in the FAISS vector database.")
    print("It shows the chunks, metadata, and how similarity search works.\n")
    
    try:
        # Load vector store
        print("Loading vector store...")
        vectorstore, embeddings, config = load_vectorstore()
        print("[OK] Vector store loaded successfully\n")
        
        # Get statistics
        total_docs = get_vectorstore_stats(vectorstore)
        print("=" * 80)
        print("VECTOR STORE STATISTICS")
        print("=" * 80)
        print(f"\nTotal Document Chunks: {total_docs}")
        print(f"Storage Location: vectorstore/")
        print("\nWhat's stored:")
        print("  • Each chunk is converted to an embedding (numerical vector)")
        print("  • Embeddings are stored in FAISS index for fast search")
        print("  • Original text chunks are stored with metadata")
        print("  • Metadata includes: source file, chunk number, etc.")
        
        # Show sample chunks
        show_sample_chunks(vectorstore, num_samples=5)
        
        # Show source breakdown
        show_source_breakdown(vectorstore)
        
        # Show embedding info
        show_embedding_info(embeddings, config)
        
        # Demonstrate similarity search
        demonstrate_similarity_search(vectorstore, embeddings)
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  • Vector store contains chunks of text from your knowledge base")
        print("  • Each chunk is converted to an embedding (vector of numbers)")
        print("  • Similar meaning = similar vectors = found together in search")
        print("  • When you ask a question:")
        print("    1. Question is converted to embedding")
        print("    2. FAISS finds similar chunks (by comparing vectors)")
        print("    3. Retrieved chunks are used as context for LLM")
        print("\nThis is the 'Retrieval' part of RAG!")
        print("\nNext step: Run code/03_build_rag.py to see how retrieval + generation work together")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nTroubleshooting:")
        print("- Ensure you've run code/02_create_vectorstore.py first")
        print("- Check that vectorstore/ directory exists")
        print("- Verify API key is set (OPENAI_API_KEY or OPENROUTER_API_KEY)")
        raise


if __name__ == "__main__":
    main()

