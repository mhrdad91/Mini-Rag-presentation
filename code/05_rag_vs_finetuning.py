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

load_dotenv()

# Check if Unsloth is available
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from datasets import Dataset
    import torch
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("[WARNING] Unsloth not installed. Install with: pip install unsloth")

# RAG imports
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
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
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
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

Context:
{context}

Question: {question}

Answer:"""
    )
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
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


def test_fine_tuned_model(model, tokenizer, question):
    """Test the fine-tuned model."""
    if model is None or tokenizer is None:
        return "Fine-tuned model not available"
    
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
    print("[WARNING] Note: Fine-tuning requires GPU and takes time.")
    print("[WARNING] For demo purposes, we'll show the concept.")
    print("\nWould you like to run fine-tuning? (y/n): ", end="")
    
    fine_tuned_model = None
    fine_tuned_tokenizer = None
    
    try:
        # For demo, we'll skip actual fine-tuning to avoid GPU requirements
        # Uncomment below if you have GPU available:
        # response = input().strip().lower()
        # if response == 'y':
        #     fine_tuned_model, fine_tuned_tokenizer = fine_tune_model(num_steps=10)
        print("Skipping fine-tuning for demo (requires GPU)...")
    except KeyboardInterrupt:
        print("\nFine-tuning skipped.")
    
    # Test RAG
    print("\n" + "="*80)
    print("OPTION 2: RAG Approach")
    print("="*80)
    
    if rag_chain:
        print("\n[OK] RAG system ready!")
        print("\nTesting RAG on sample questions:\n")
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. Question: {question}")
            try:
                answer = rag_chain.invoke(question)
                print(f"   RAG Answer: {answer[:200]}...")
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

