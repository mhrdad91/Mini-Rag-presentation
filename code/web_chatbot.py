"""
Interactive Web Chatbot for RAG Presentation
============================================

A simple Streamlit web interface for the RAG system.
Attendees can ask questions and get answers based on the knowledge base.

Run with:
    streamlit run code/web_chatbot.py

Or use the run script:
    python run_web_demo.py
"""

import os
import sys
from pathlib import Path
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_config import get_api_config, get_embedding_model, get_llm_model

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="TechCorp RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #1565a0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    """Load the RAG system (cached for performance)."""
    try:
        # Check API key
        config = get_api_config()
        if not config:
            return None, "API key not found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file."
        
        # Load vector store
        vectorstore_path = Path("vectorstore")
        if not vectorstore_path.exists():
            return None, "Vector store not found. Please run code/02_create_vectorstore.py first."
        
        # Setup embeddings
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
        
        # Setup retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # More chunks for fun content
        
        # Setup LLM
        llm_model_name = get_llm_model(config["provider"])
        llm_kwargs = {
            "model": llm_model_name,
            "temperature": 0.3,  # Slightly higher for more complete answers
            "openai_api_key": config["api_key"]
        }
        if config["base_url"]:
            llm_kwargs["openai_api_base"] = config["base_url"]
        
        llm = ChatOpenAI(**llm_kwargs)
        
        # Setup prompt
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
        
        # Format docs function
        def format_docs(docs):
            return "\n\n".join(
                f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
                for doc in docs
            )
        
        # Build RAG chain
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, retriever, config["provider"]
        
    except Exception as e:
        return None, f"Error loading RAG system: {str(e)}"


def get_sources(retriever, question):
    """Get source documents for a question."""
    try:
        docs = retriever.get_relevant_documents(question)
        sources = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            source_name = Path(source).name if source != 'Unknown' else 'Unknown'
            sources.append(source_name)
        return sources
    except Exception as e:
        return []


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<p class="main-header">🤖 TechCorp Customer Support Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask me anything about TechCorp products, pricing, or support!</p>', unsafe_allow_html=True)
    
    # Load RAG system
    rag_result = load_rag_system()
    
    if rag_result[0] is None:
        st.error(rag_result[1])
        st.info("**Setup Instructions:**\n1. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env file\n2. Run: `python code/02_create_vectorstore.py`")
        return
    
    rag_chain, retriever, provider = rag_result
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(f"**Provider:** {provider.upper()}\n\n**Status:** ✅ Ready\n\n**Knowledge Base:** 5 documents loaded")
        
        st.header("💡 Sample Questions")
        sample_questions = [
            "How do I reset my password?",
            "What are the pricing plans?",
            "How do I cancel my subscription?",
            "What is your refund policy?",
            "How do I add team members?",
            "What is data privacy?",
        ]
        
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state.question = q
        
        st.header("📚 Documentation")
        st.markdown("""
        **RAG System:**
        - Retrieves relevant information
        - Generates answers using LLM
        - Cites sources
        
        **Built with:**
        - LangChain
        - FAISS Vector Store
        - OpenAI-compatible API
        """)
    
    # Main chat interface
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📄 Sources"):
                    for i, source in enumerate(set(message["sources"]), 1):
                        st.write(f"{i}. {source}")
    
    # Chat input
    question = st.chat_input("Ask a question about TechCorp...")
    
    # Handle question from input or sidebar button
    if "question" in st.session_state:
        question = st.session_state.question
        del st.session_state.question
    
    if question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base and generating answer..."):
                try:
                    # Get sources
                    sources = get_sources(retriever, question)
                    
                    # Get answer
                    answer = rag_chain.invoke(question)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📄 Sources"):
                            for i, source in enumerate(set(sources), 1):
                                st.write(f"{i}. {source}")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()

