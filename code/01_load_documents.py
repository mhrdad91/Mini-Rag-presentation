"""
Step 1: Load and Prepare Documents

This script demonstrates how to load documents from a knowledge base
and prepare them for RAG processing by splitting them into chunks.

Key concepts:
- Document loading from directory
- Text splitting strategies
- Chunk overlap for context preservation
- Metadata for tracking sources
"""

import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_documents():
    """Load documents from the knowledge_base directory."""
    knowledge_base_path = Path("knowledge_base")
    
    if not knowledge_base_path.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found at {knowledge_base_path}. "
            "Please ensure the knowledge_base directory exists with markdown files."
        )
    
    loader = DirectoryLoader(
        path=str(knowledge_base_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True
    )
    
    documents = loader.load()
    
    print(f"\n✅ Loaded {len(documents)} documents from knowledge base")
    print(f"📄 Document sources:")
    for doc in documents:
        print(f"   - {doc.metadata.get('source', 'Unknown')}")
    
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of Document objects
        chunk_size: Target size for each chunk (in characters)
        chunk_overlap: Overlap between chunks to preserve context
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Try these separators in order
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"\n✅ Split documents into {len(chunks)} chunks")
    print(f"📊 Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks):.0f} characters")
    print(f"📊 Chunk size range: {min(len(chunk.page_content) for chunk in chunks)} - {max(len(chunk.page_content) for chunk in chunks)} characters")
    
    # Show example chunk
    print(f"\n📝 Example chunk:")
    print(f"   Source: {chunks[0].metadata.get('source', 'Unknown')}")
    print(f"   Length: {len(chunks[0].page_content)} characters")
    print(f"   Content preview: {chunks[0].page_content[:200]}...")
    
    return chunks


def main():
    """Main function to load and prepare documents."""
    print("=" * 80)
    print("STEP 1: Loading and Preparing Documents")
    print("=" * 80)
    
    try:
        # Load documents
        documents = load_documents()
        
        # Split documents into chunks
        chunks = split_documents(
            documents,
            chunk_size=1000,    # Adjust based on your needs
            chunk_overlap=200   # Overlap helps preserve context
        )
        
        print("\n" + "=" * 80)
        print("✅ Step 1 Complete!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run: python code/02_create_vectorstore.py")
        print("   This will create embeddings and store them in a vector database")
        
        return chunks
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Ensure knowledge_base directory exists with .md files")
        print("- Check that you have necessary permissions")
        print("- Verify langchain-community is installed")
        raise


if __name__ == "__main__":
    chunks = main()

