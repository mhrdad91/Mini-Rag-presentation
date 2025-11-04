#!/usr/bin/env python3
"""
Complete RAG Presentation Test Script
====================================

This script runs all steps of the RAG presentation in sequence
to verify everything works correctly.

Usage:
    python run_all.py

It will:
1. Check dependencies
2. Run all code steps in order
3. Test the final chatbot
4. Optionally run comparison demos
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Load environment variables
load_dotenv()

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_step(step_num, total_steps, description):
    """Print a step indicator."""
    print(f"{Colors.OKCYAN}[{step_num}/{total_steps}]{Colors.ENDC} {Colors.BOLD}{description}{Colors.ENDC}")


def print_success(message):
    """Print a success message."""
    print(f"{Colors.OKGREEN}[OK] {message}{Colors.ENDC}")


def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.WARNING}[WARNING] {message}{Colors.ENDC}")


def print_error(message):
    """Print an error message."""
    print(f"{Colors.FAIL}[ERROR] {message}{Colors.ENDC}")


def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("Checking Dependencies")
    
    required_packages = [
        'langchain',
        'langchain_openai',
        'langchain_community',
        'faiss',
        'dotenv',
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'faiss':
                __import__('faiss')
            elif package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            missing.append(package)
            print_error(f"{package} is not installed")
    
    if missing:
        print_error(f"\nMissing packages: {', '.join(missing)}")
        print_warning("Install with: pip install -r requirements.txt")
        return False
    
    print_success("All required dependencies are installed!")
    return True


def check_environment():
    """Check environment variables."""
    print_header("Checking Environment")
    
    # Check for OpenRouter first (preferred), then OpenAI
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if openrouter_key:
        masked_key = openrouter_key[:8] + "..." + openrouter_key[-4:] if len(openrouter_key) > 12 else "***"
        print_success(f"OPENROUTER_API_KEY is set ({masked_key})")
        print_success("Using OpenRouter API (supports embeddings and LLM)")
        return True
    elif openai_key:
        masked_key = openai_key[:8] + "..." + openai_key[-4:] if len(openai_key) > 12 else "***"
        print_success(f"OPENAI_API_KEY is set ({masked_key})")
        print_success("Using OpenAI API")
        return True
    else:
        print_error("No API key found!")
        print_warning("Please set one of:")
        print_warning("  - OPENROUTER_API_KEY (recommended - get from https://openrouter.ai/keys)")
        print_warning("  - OPENAI_API_KEY (get from https://platform.openai.com/api-keys)")
        print_warning("\nCreate a .env file with one of these keys")
        return False


def check_knowledge_base():
    """Check if knowledge base exists."""
    print_header("Checking Knowledge Base")
    
    kb_path = Path("knowledge_base")
    if not kb_path.exists():
        print_error("knowledge_base directory not found!")
        return False
    
    md_files = list(kb_path.glob("*.md"))
    if not md_files:
        print_error("No markdown files found in knowledge_base!")
        return False
    
    print_success(f"Found {len(md_files)} knowledge base files:")
    for file in md_files:
        print(f"  • {file.name}")
    
    return True


def run_script(script_path, description, required=True):
    """Run a Python script and return success status."""
    script = Path(script_path)
    
    if not script.exists():
        if required:
            print_error(f"Script not found: {script_path}")
            return False
        else:
            print_warning(f"Optional script not found: {script_path}")
            return True
    
    print(f"\n{Colors.OKBLUE}Running: {script.name}{Colors.ENDC}")
    print(f"Description: {description}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print_success(f"{script.name} completed successfully")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 10:
                    print(f"{Colors.OKCYAN}... (showing last 10 lines){Colors.ENDC}")
                    for line in lines[-10:]:
                        print(f"  {line}")
                else:
                    for line in lines:
                        print(f"  {line}")
            return True
        else:
            print_error(f"{script.name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"{Colors.FAIL}Error output:{Colors.ENDC}")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print_error(f"{script.name} timed out after 5 minutes")
        return False
    except Exception as e:
        print_error(f"Error running {script.name}: {e}")
        return False


def check_vectorstore():
    """Check if vector store exists."""
    vectorstore_path = Path("vectorstore")
    if vectorstore_path.exists():
        files = list(vectorstore_path.iterdir())
        if files:
            print_success(f"Vector store exists ({len(files)} files)")
            return True
    
    print_warning("Vector store not found")
    return False


def main():
    """Main function to run all steps."""
    print_header("RAG Presentation Complete Test Suite")
    
    print(f"{Colors.BOLD}This script will test all components of the RAG presentation.{Colors.ENDC}\n")
    
    # Track steps
    steps_completed = 0
    steps_failed = 0
    
    # Step 1: Check dependencies
    print_step(1, 7, "Checking Dependencies")
    if not check_dependencies():
        print_error("\nPlease install dependencies first!")
        return False
    steps_completed += 1
    
    # Step 2: Check environment
    print_step(2, 7, "Checking Environment")
    if not check_environment():
        print_error("\nPlease set up your environment first!")
        return False
    steps_completed += 1
    
    # Step 3: Check knowledge base
    print_step(3, 7, "Checking Knowledge Base")
    if not check_knowledge_base():
        print_error("\nKnowledge base files are missing!")
        return False
    steps_completed += 1
    
    # Step 4: Load documents
    print_step(4, 7, "Step 1: Loading Documents")
    if run_script("code/01_load_documents.py", "Load and split documents"):
        steps_completed += 1
    else:
        steps_failed += 1
        print_error("Failed at document loading step!")
        return False
    
    # Step 5: Create vector store
    print_step(5, 7, "Step 2: Creating Vector Store")
    if run_script("code/02_create_vectorstore.py", "Create embeddings and vector store"):
        steps_completed += 1
    else:
        steps_failed += 1
        print_error("Failed at vector store creation step!")
        return False
    
    # Verify vector store was created
    if not check_vectorstore():
        print_error("Vector store was not created successfully!")
        return False
    
    # Step 6: Build RAG system
    print_step(6, 7, "Step 3: Building RAG System")
    if run_script("code/03_build_rag.py", "Build complete RAG chain"):
        steps_completed += 1
    else:
        steps_failed += 1
        print_error("Failed at RAG system building step!")
        return False
    
    # Step 7: Test chatbot (quick test)
    print_step(7, 7, "Step 4: Testing Chatbot")
    print_warning("Testing chatbot with a sample question...")
    
    # Quick test instead of full interactive session
    test_script = """
import os
import sys
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for utils
project_root = Path(os.getcwd())
sys.path.insert(0, str(project_root))
from utils.api_config import get_api_config, get_embedding_model, get_llm_model

vectorstore_path = project_root / "vectorstore"
config = get_api_config()
if not config:
    print("[ERROR] API key not found!")
    sys.exit(1)

model_name = get_embedding_model(config["provider"])
embedding_kwargs = {
    "model": model_name,
    "openai_api_key": config["api_key"]
}
if config["base_url"]:
    embedding_kwargs["openai_api_base"] = config["base_url"]

embeddings = OpenAIEmbeddings(**embedding_kwargs)
vectorstore = FAISS.load_local(str(vectorstore_path), embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

prompt_template = ChatPromptTemplate.from_template(
    "You are a helpful customer support assistant.\\n\\nContext:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:"
)

model_name = get_llm_model(config["provider"])
llm_kwargs = {
    "model": model_name,
    "temperature": 0,
    "openai_api_key": config["api_key"]
}
if config["base_url"]:
    llm_kwargs["openai_api_base"] = config["base_url"]

llm = ChatOpenAI(**llm_kwargs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

test_question = "How do I reset my password?"
print(f"Testing question: {test_question}")
result = rag_chain.invoke(test_question)
print(f"Answer: {result[:200]}...")
print("[OK] Chatbot test successful!")
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print_success("Chatbot test passed!")
            print(result.stdout)
            steps_completed += 1
        else:
            print_error("Chatbot test failed!")
            print(result.stderr)
            steps_failed += 1
    except Exception as e:
        print_error(f"Error testing chatbot: {e}")
        steps_failed += 1
    
    # Summary
    print_header("Test Summary")
    
    print(f"{Colors.BOLD}Steps Completed: {steps_completed}/7{Colors.ENDC}")
    if steps_failed > 0:
        print(f"{Colors.FAIL}Steps Failed: {steps_failed}{Colors.ENDC}")
    else:
        print(f"{Colors.OKGREEN}All steps completed successfully!{Colors.ENDC}")
    
    # Optional demos
    print(f"\n{Colors.OKCYAN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}Optional Demos:{Colors.ENDC}\n")
    
    print("1. Interactive Chatbot:")
    print(f"   {Colors.OKBLUE}python code/04_chatbot.py{Colors.ENDC}")
    print()
    
    print("2. RAG vs Fine-Tuning Comparison:")
    print(f"   {Colors.OKBLUE}python code/05_rag_vs_finetuning.py{Colors.ENDC}")
    print()
    
    print("3. Fine-Tuning Demo (requires GPU):")
    print(f"   {Colors.OKBLUE}python code/06_finetune_unsloth.py{Colors.ENDC}")
    print()
    
    if steps_failed == 0:
        print(f"{Colors.OKGREEN}{Colors.BOLD}[SUCCESS] All tests passed! Your RAG system is ready to use.{Colors.ENDC}\n")
        return True
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}[FAILED] Some tests failed. Please check the errors above.{Colors.ENDC}\n")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}\nTest interrupted by user.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

