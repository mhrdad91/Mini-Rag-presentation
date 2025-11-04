"""
Step 3: Build the RAG System

This script demonstrates how to build a complete RAG (Retrieval-Augmented Generation)
system using LangChain.

Key concepts:
- Retriever: Finds relevant document chunks
- Prompt template: Formats question + context for LLM
- RAG chain: Combines retrieval + generation
- LangChain's chain composition makes this easy!
"""

import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

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
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print("✅ Vector store loaded successfully")
    return vectorstore


def create_retriever(vectorstore, top_k=3):
    """
    Create a retriever from the vector store.
    
    Args:
        vectorstore: FAISS vector store
        top_k: Number of relevant chunks to retrieve
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    
    print(f"✅ Retriever created (retrieving top {top_k} chunks)")
    return retriever


def create_prompt_template():
    """Create a prompt template for the RAG system."""
    template = """You are a helpful customer support assistant for TechCorp.
Your job is to answer customer questions based on the provided context.

Use the following pieces of context to answer the question.
If you don't know the answer based on the context, just say that you don't know.
Don't make up information. Be concise and helpful.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    print("✅ Prompt template created")
    return prompt


def format_docs(docs):
    """Format retrieved documents for the prompt."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
        for doc in docs
    )


def build_rag_chain(retriever, prompt_template, llm):
    """
    Build the RAG chain that combines retrieval and generation.
    
    The chain:
    1. Takes a question
    2. Retrieves relevant chunks
    3. Formats them with the prompt
    4. Sends to LLM
    5. Returns the answer
    """
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    print("✅ RAG chain built successfully")
    return rag_chain


def test_rag_system(rag_chain):
    """Test the RAG system with sample questions."""
    test_questions = [
        "How do I reset my password?",
        "What are the pricing plans?",
        "How do I cancel my subscription?",
    ]
    
    print("\n" + "=" * 80)
    print("Testing RAG System")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 80)
        
        try:
            answer = rag_chain.invoke(question)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to build the RAG system."""
    print("=" * 80)
    print("STEP 3: Building the RAG System")
    print("=" * 80)
    
    try:
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file."
            )
        
        # Load vector store
        vectorstore = load_vectorstore()
        
        # Create retriever
        retriever = create_retriever(vectorstore, top_k=3)
        
        # Create prompt template
        prompt_template = create_prompt_template()
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Cost-effective model
            temperature=0,  # Low temperature for factual answers
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        print("✅ LLM initialized")
        
        # Build RAG chain
        rag_chain = build_rag_chain(retriever, prompt_template, llm)
        
        # Test the system
        test_rag_system(rag_chain)
        
        print("\n" + "=" * 80)
        print("✅ Step 3 Complete!")
        print("=" * 80)
        print("\nKey concepts:")
        print("• Retriever finds relevant chunks from vector store")
        print("• Prompt template formats question + context")
        print("• LLM generates answer using context")
        print("• LangChain chains make composition easy!")
        print("\nNext steps:")
        print("1. Run: python code/04_chatbot.py")
        print("   This will create an interactive chatbot interface")
        
        return rag_chain
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Ensure vector store exists (run Step 2 first)")
        print("- Check OPENAI_API_KEY is set correctly")
        print("- Verify you have OpenAI API credits")
        raise


if __name__ == "__main__":
    rag_chain = main()

