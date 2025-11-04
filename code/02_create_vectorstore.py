"""
Step 2: Create Embeddings and Vector Store

This script demonstrates how to:
1. Create embeddings for document chunks
2. Store embeddings in a vector database (FAISS)
3. Save the vector store for reuse

Key concepts:
- Embeddings convert text to numerical vectors
- Vector databases enable fast similarity search
- FAISS is a popular open-source vector database
"""

import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import sys

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_config import get_api_config, get_embedding_model

# Load environment variables
load_dotenv()

def check_api_key():
    """Check if API key is set (OpenAI or OpenRouter)."""
    config = get_api_config()
    if not config:
        raise ValueError(
            "API key not found! Please set one of:\n"
            "  - OPENROUTER_API_KEY (for OpenRouter)\n"
            "  - OPENAI_API_KEY (for OpenAI)\n\n"
            "Get keys from:\n"
            "  - OpenRouter: https://openrouter.ai/keys\n"
            "  - OpenAI: https://platform.openai.com/api-keys"
        )
    
    provider = config["provider"]
    print(f"[OK] {provider.upper()} API key found")
    return config


def load_and_split_documents():
    """Load and split documents (reusing Step 1 logic)."""
    knowledge_base_path = Path("knowledge_base")
    
    if not knowledge_base_path.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found at {knowledge_base_path}"
        )
    
    loader = DirectoryLoader(
        path=str(knowledge_base_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True
    )
    
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"[OK] Loaded and split {len(chunks)} document chunks")
    return chunks


def create_embeddings_and_vectorstore(chunks, config):
    """
    Create embeddings and store them in FAISS vector database.
    
    Args:
        chunks: List of Document chunks to embed
        config: API configuration dict
    """
    print("\nCreating embeddings...")
    print("   This may take a moment depending on the number of chunks...")
    
    # Get embedding model name
    model_name = get_embedding_model(config["provider"])
    
    # Initialize embedding model
    # Supports both OpenAI and OpenRouter (OpenAI-compatible)
    embedding_kwargs = {
        "model": model_name,
        "openai_api_key": config["api_key"]
    }
    
    # Add base_url for OpenRouter
    if config["base_url"]:
        embedding_kwargs["openai_api_base"] = config["base_url"]
    
    embeddings = OpenAIEmbeddings(**embedding_kwargs)
    
    print(f"   Using model: {model_name}")
    print(f"   Provider: {config['provider'].upper()}")
    
    # Create vector store from documents
    # This will:
    # 1. Generate embeddings for each chunk
    # 2. Store them in FAISS index
    # 3. Save to disk for reuse
    vectorstore_path = Path("vectorstore")
    vectorstore_path.mkdir(exist_ok=True)
    
    print(f"\nCreating vector store...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Save vector store to disk
    save_path = str(vectorstore_path)
    vectorstore.save_local(save_path)
    
    print(f"[OK] Vector store saved to: {save_path}")
    
    # Test the vector store with a sample query
    print(f"\nTesting vector store with sample query...")
    test_query = "How do I reset my password?"
    results = vectorstore.similarity_search(test_query, k=2)
    
    print(f"   Query: '{test_query}'")
    print(f"   Found {len(results)} relevant chunks:")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        preview = doc.page_content[:150].replace('\n', ' ')
        print(f"   {i}. Source: {Path(source).name}")
        print(f"      Preview: {preview}...")
    
    return vectorstore


def main():
    """Main function to create embeddings and vector store."""
    print("=" * 80)
    print("STEP 2: Creating Embeddings and Vector Store")
    print("=" * 80)
    
    try:
        # Check API key
        config = check_api_key()
        
        # Load and split documents
        chunks = load_and_split_documents()
        
        # Create embeddings and vector store
        vectorstore = create_embeddings_and_vectorstore(chunks, config)
        
        print("\n" + "=" * 80)
        print("[OK] Step 2 Complete!")
        print("=" * 80)
        print("\nKey concepts:")
        print("• Embeddings convert text → numerical vectors")
        print("• Similar meaning = similar vectors")
        print("• Vector store enables fast similarity search")
        print("• Vector store saved locally for reuse")
        print("\nNext steps:")
        print("1. Run: python code/03_build_rag.py")
        print("   This will build the complete RAG system")
        print("\nNote: We'll dive deeper into embeddings at the end of the presentation!")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nTroubleshooting:")
        print("- Ensure OPENAI_API_KEY or OPENROUTER_API_KEY is set in .env file")
        print("- Check your API key is valid")
        print("- Verify you have credits/quota in your account")
        print("- Run Step 1 first if you haven't already")
        raise


if __name__ == "__main__":
    vectorstore = main()

