"""
RAG vs Fine-Tuning Demo
=======================

This script demonstrates the difference between RAG and Fine-Tuning approaches
using a small 1B model (TinyLlama) with Unsloth.

We'll compare:
1. Fine-Tuned Model: Trained on customer support data
2. RAG System: Uses retrieval from knowledge base

Same questions, different approaches!
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

load_dotenv()

import platform

# Check platform
APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Check if Unsloth is available (for NVIDIA/AMD GPUs)
UNSLOTH_AVAILABLE = False
try:
    if not APPLE_SILICON:  # Only try to import on non-Apple Silicon
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from datasets import Dataset
        import torch
        UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError):
    UNSLOTH_AVAILABLE = False

# Check if MLX is available (for Apple Silicon)
MLX_AVAILABLE = False
MLX_LM_AVAILABLE = False
try:
    if APPLE_SILICON:
        import mlx.core as mx
        import mlx.nn as nn
        MLX_AVAILABLE = True
        # Check for mlx_lm (try importing the module)
        try:
            import mlx_lm
            MLX_LM_AVAILABLE = True
        except ImportError:
            MLX_LM_AVAILABLE = False
except ImportError:
    MLX_AVAILABLE = False
    MLX_LM_AVAILABLE = False

# RAG imports
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    import sys
    from pathlib import Path
    
    # Add parent directory to path for utils
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.api_config import get_api_config, get_embedding_model, get_llm_model
    
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("[WARNING] LangChain not installed. Install with: pip install langchain langchain-openai langchain-community faiss-cpu")


def create_training_data():
    """
    Create a small training dataset for fine-tuning.
    This simulates customer support Q&A pairs.
    """
    training_data = [
        {
            "instruction": "How do I reset my password?",
            "output": "To reset your password, go to the login page, click 'Forgot Password', enter your email address, and follow the instructions in the email you receive."
        },
        {
            "instruction": "What are the pricing plans?",
            "output": "We offer Free ($0/month), Starter ($10/user/month), Professional ($25/user/month), and Enterprise (custom pricing). The Free plan includes 3 projects and 5 team members."
        },
        {
            "instruction": "How do I cancel my subscription?",
            "output": "Go to Account Settings > Billing and click 'Cancel Subscription'. Your account remains active until the end of your billing period."
        },
        {
            "instruction": "Can I export my data?",
            "output": "Yes, go to Account Settings > Privacy & Data and click 'Request Data Export'. You can export in CSV, JSON, or Excel format."
        },
        {
            "instruction": "How many team members can I invite?",
            "output": "Free plan: 5 members, Starter: 25 members, Professional and Enterprise: Unlimited members."
        },
        {
            "instruction": "What happens if I cancel?",
            "output": "Your account remains active until the end of your billing period. After that, you'll be moved to the Free plan or your account will be deactivated. Data is retained for 30 days."
        },
        {
            "instruction": "Do you offer refunds?",
            "output": "Refunds are available for annual plans on a prorated basis. Monthly plans are non-refundable."
        },
        {
            "instruction": "What browsers are supported?",
            "output": "We support Chrome, Firefox, Safari, and Edge (latest versions)."
        },
        {
            "instruction": "Is there a mobile app?",
            "output": "Yes, we have iOS and Android apps available in the App Store and Google Play Store."
        },
        {
            "instruction": "How do I invite team members?",
            "output": "Navigate to 'Team' in the sidebar, click 'Invite Members', enter email addresses, select their role, and click 'Send Invitations'."
        }
    ]
    
    return training_data


def format_training_data(data):
    """Format data for Unsloth fine-tuning."""
    formatted = []
    for item in data:
        formatted.append({
            "text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        })
    return formatted


def setup_rag_system():
    """Setup RAG system for comparison."""
    if not RAG_AVAILABLE:
        return None
    
    vectorstore_path = Path("vectorstore")
    if not vectorstore_path.exists():
        print("[WARNING] Vector store not found. Run code/02_create_vectorstore.py first.")
        return None
    
    config = get_api_config()
    if not config:
        print("[ERROR] API key not found. Set OPENAI_API_KEY or OPENROUTER_API_KEY")
        return None
    
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
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt_template = ChatPromptTemplate.from_template(
        """You are a helpful customer support assistant for TechCorp.
Answer based on the provided context.

IMPORTANT: The context contains our official company policies and procedures. Always use these official methods as the primary answer. These are our established procedures, not alternatives - they represent how TechCorp actually operates.

Context:
{context}

Question: {question}

Answer:"""
    )
    
    config = get_api_config()
    if not config:
        print("[ERROR] API key not found. Set OPENAI_API_KEY or OPENROUTER_API_KEY")
        return None
    
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
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def fine_tune_model(model_name="unsloth/tinyllama-bnb-4bit", num_steps=10):
    """
    Fine-tune a small model using Unsloth.
    
    Note: This is a minimal example. For production, use more steps and data.
    """
    if not UNSLOTH_AVAILABLE:
        print("[ERROR] Unsloth not available. Cannot fine-tune.")
        return None
    
    print(f"\n{'='*80}")
    print("FINE-TUNING WITH UNSLOTH")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Steps: {num_steps}")
    print("\nLoading model...")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        use_gradient_checkpointing=True,
        random_state=3407,
    )
    
    # Prepare training data
    print("\nPreparing training data...")
    training_data = create_training_data()
    formatted_data = format_training_data(training_data)
    dataset = Dataset.from_list(formatted_data)
    
    # Setup trainer
    print("\nStarting fine-tuning...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=2048,
        dataset_text_field="text",
        packing=False,
    )
    
    # Train (minimal steps for demo)
    trainer.train()
    
    # Fast inference mode
    FastLanguageModel.for_inference(model)
    
    print("[OK] Fine-tuning complete!")
    
    return model, tokenizer


def test_fine_tuned_model_mlx(adapter_path, question):
    """Test the fine-tuned MLX model."""
    if not adapter_path or not Path(adapter_path).exists():
        return "Fine-tuned adapter not available"
    
    try:
        import subprocess
        model_name = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        
        # Use mlx_lm generate command
        cmd = [
            "python", "-m", "mlx_lm", "generate",
            "--model", model_name,
            "--adapter-path", adapter_path,
            "--prompt", question,
            "--max-tokens", "200",
            "--temp", "0.7"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Extract response (mlx_lm outputs the full prompt + response)
            output = result.stdout
            if "Answer:" in output or "Response:" in output:
                # Try to extract just the response part
                lines = output.split("\n")
                response_lines = []
                capturing = False
                for line in lines:
                    if "Answer:" in line or "Response:" in line:
                        capturing = True
                        response_lines.append(line.split(":")[-1].strip())
                    elif capturing and line.strip():
                        response_lines.append(line.strip())
                
                if response_lines:
                    return " ".join(response_lines)
            return output[-500:] if len(output) > 500 else output  # Last 500 chars
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error testing fine-tuned model: {str(e)}"


def test_fine_tuned_model(model, tokenizer, question):
    """Test the fine-tuned model (Unsloth version)."""
    if model is None or tokenizer is None:
        return "Fine-tuned model not available"
    
    try:
        import torch
        # Format prompt
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"


def compare_approaches():
    """Compare RAG vs Fine-Tuning on the same questions."""
    
    test_questions = [
        "How do I reset my password?",
        "What are the pricing plans?",
        "How do I cancel my subscription?",
    ]
    
    print(f"\n{'='*80}")
    print("RAG vs FINE-TUNING COMPARISON")
    print(f"{'='*80}\n")
    
    # Setup RAG
    rag_chain = setup_rag_system()
    
    # Fine-tune model (quick demo)
    print("\n" + "="*80)
    print("OPTION 1: Fine-Tuning Approach")
    print("="*80)
    
    fine_tuned_model = None
    fine_tuned_tokenizer = None
    fine_tuned_adapter_path = None
    
    if APPLE_SILICON:
        print("[OK] Detected Apple Silicon Mac")
        if MLX_LM_AVAILABLE:
            print("[OK] MLX framework available")
            # Check if fine-tuned adapter exists
            adapter_path = Path("adapters/techcorp-support")
            if adapter_path.exists():
                print(f"[OK] Found fine-tuned adapter at: {adapter_path}")
                fine_tuned_adapter_path = str(adapter_path)
                print("[INFO] Fine-tuned model available for comparison!")
            else:
                print("[INFO] Fine-tuned adapter not found.")
                print("[INFO] To create one, run:")
                print("       python code/08_finetune_mlx_complete.py")
                print("[INFO] For now, we'll show the comparison concept.")
        elif MLX_AVAILABLE:
            print("[INFO] MLX core available, but mlx_lm not found")
            print("[INFO] Install mlx_lm: pip install mlx-lm")
            print("[INFO] Then run: python code/08_finetune_mlx_complete.py")
        else:
            print("[INFO] Install MLX for fine-tuning on Apple Silicon:")
            print("       pip install mlx mlx-lm transformers datasets")
            print("       Then run: python code/08_finetune_mlx_complete.py")
    
    elif UNSLOTH_AVAILABLE:
        print("[OK] Unsloth available (NVIDIA/AMD GPU detected)")
        print("[WARNING] Note: Fine-tuning requires GPU and takes time.")
        print("[WARNING] For demo purposes, we'll show the concept.")
        print("[INFO] To actually fine-tune, run: python code/06_finetune_unsloth.py")
    else:
        print("[INFO] Unsloth not available (requires NVIDIA/AMD GPU)")
        print("[INFO] Install with: pip install unsloth")
        print("[INFO] Or use MLX on Apple Silicon: pip install mlx mlx-lm")
    
    print("\n[INFO] For this demo, we'll focus on RAG vs Fine-Tuning concepts.")
    print("[INFO] Actual fine-tuning comparison requires running the fine-tuning scripts separately.")
    
    # Test RAG
    print("\n" + "="*80)
    print("OPTION 2: RAG Approach")
    print("="*80)
    
    if rag_chain:
        print("\n[OK] RAG system ready!")
        print("\nTesting RAG on sample questions:\n")
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. Question: {question}")
            print("-" * 80)
            try:
                answer = rag_chain.invoke(question)
                print(f"   RAG Answer: {answer[:300]}...")
            except Exception as e:
                print(f"   Error: {e}")
            
            # Test fine-tuned model if available (MLX)
            if fine_tuned_adapter_path and APPLE_SILICON:
                print("\n   Fine-Tuned Model (MLX) Answer:")
                try:
                    ft_answer = test_fine_tuned_model_mlx(fine_tuned_adapter_path, question)
                    print(f"   {ft_answer[:300]}...")
                except Exception as e:
                    print(f"   Error: {e}")
            
            print()
    else:
        print("[ERROR] RAG system not available")
    
    # Comparison
    print("\n" + "="*80)
    print("KEY DIFFERENCES")
    print("="*80)
    print("""
Fine-Tuning:
  [+] Model learns domain-specific patterns
  [+] No retrieval step needed at inference
  [-] Requires training time (hours)
  [-] Needs GPU for training
  [-] Hard to update (must retrain)
  [-] Limited by training data

RAG:
  [+] No training needed
  [+] Easy to update (just change knowledge base)
  [+] Works with any LLM
  [+] Can cite sources
  [+] Handles dynamic information
  [-] Requires retrieval step (slight latency)
  [-] Needs vector database setup
    """)


if __name__ == "__main__":
    compare_approaches()

