"""
Step 4: Interactive Chatbot

This script creates an interactive chatbot interface for the RAG system.
Users can ask questions and get answers based on the knowledge base.

Features:
- Interactive question-answer loop
- Source citation for answers
- Graceful error handling
- Clean exit with 'quit' or 'exit'
"""

import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import sys

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_config import get_api_config, get_embedding_model, get_llm_model

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
    
    return vectorstore


def format_docs(docs):
    """Format retrieved documents for the prompt."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
        for doc in docs
    )


def build_rag_chain(vectorstore):
    """Build the RAG chain."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Increased to get more context including fun methods
    )
    
    prompt_template = ChatPromptTemplate.from_template(
        """You are a helpful customer support assistant for TechCorp.
Your job is to answer customer questions based on the provided context.

Use the following pieces of context to answer the question.
If you don't know the answer based on the context, just say that you don't know.
Don't make up information. Be concise and helpful.

IMPORTANT: If the context mentions both standard methods AND fun/alternative methods, 
include BOTH in your answer. The fun methods are part of our unique culture and should be shared!

Context:
{context}

Question: {question}

Answer:"""
    )
    
    config = get_api_config()
    if not config:
        raise ValueError("API key not found. Set OPENAI_API_KEY or OPENROUTER_API_KEY")
    
    model_name = get_llm_model(config["provider"])
    
    llm_kwargs = {
        "model": model_name,
        "temperature": 0.3,  # Slightly higher for more complete answers
        "openai_api_key": config["api_key"]
    }
    
    if config["base_url"]:
        llm_kwargs["openai_api_base"] = config["base_url"]
    
    llm = ChatOpenAI(**llm_kwargs)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


def get_sources(retriever, question):
    """Get source documents for a question."""
    docs = retriever.get_relevant_documents(question)
    sources = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        source_name = Path(source).name if source != 'Unknown' else 'Unknown'
        sources.append(source_name)
    return sources


def main():
    """Main function to run the interactive chatbot."""
    print("=" * 80)
    print("TechCorp Customer Support Chatbot")
    print("=" * 80)
    print("\nAsk me anything about TechCorp products, pricing, or support!")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    try:
        # Check API key
        config = get_api_config()
        if not config:
            raise ValueError(
                "API key not found. Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file."
            )
        print(f"[OK] Using {config['provider'].upper()} API")
        
        # Load vector store
        print("Loading knowledge base...")
        vectorstore = load_vectorstore()
        
        # Build RAG chain
        print("Initializing chatbot...")
        rag_chain, retriever = build_rag_chain(vectorstore)
        
        print("[OK] Ready! Ask me anything.\n")
        print("-" * 80)
        
        # Interactive loop
        while True:
            try:
                # Get user question
                question = input("\nYour question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nThanks for using TechCorp Support Chatbot!")
                    break
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                # Get sources
                print("\nSearching knowledge base...")
                sources = get_sources(retriever, question)
                
                # Get answer
                print("Generating answer...")
                answer = rag_chain.invoke(question)
                
                # Display answer
                print("\n" + "=" * 80)
                print("Answer:")
                print("-" * 80)
                print(answer)
                print("-" * 80)
                
                # Display sources
                if sources:
                    print(f"\nSources:")
                    for i, source in enumerate(set(sources), 1):
                        print(f"   {i}. {source}")
                
                print("=" * 80)
                
            except KeyboardInterrupt:
                print("\n\nThanks for using TechCorp Support Chatbot!")
                break
            except Exception as e:
                print(f"\n[ERROR] Error: {e}")
                print("Please try again or type 'quit' to exit.")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nTroubleshooting:")
        print("- Ensure vector store exists (run Step 2 first)")
        print("- Check OPENAI_API_KEY or OPENROUTER_API_KEY is set correctly")
        print("- Verify you have API credits")
        raise


if __name__ == "__main__":
    main()

