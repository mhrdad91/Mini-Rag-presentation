"""
Complete Fine-Tuning Demo with Unsloth
=======================================

This script provides a complete example of fine-tuning a 1B model
using Unsloth for customer support tasks.

Requirements:
- GPU with at least 8GB VRAM
- CUDA drivers installed
- Unsloth installed: pip install unsloth
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from datasets import Dataset
    import torch
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("❌ Unsloth not installed!")
    print("Install with: pip install unsloth")
    exit(1)


def create_training_dataset():
    """
    Create training dataset for customer support fine-tuning.
    In production, you'd load this from a larger dataset.
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
        },
        {
            "instruction": "What is the Free plan limit?",
            "output": "The Free plan includes up to 3 projects and 5 team members with basic task management features."
        },
        {
            "instruction": "How do I change my email address?",
            "output": "Go to Account Settings > Profile and update your email. You'll need to verify the new email address."
        },
        {
            "instruction": "Can I use TechCorp offline?",
            "output": "Limited offline functionality is available in mobile apps. Full offline mode is coming soon."
        },
        {
            "instruction": "What payment methods do you accept?",
            "output": "We accept credit cards (Visa, MasterCard, American Express), debit cards, PayPal, and bank transfers (Enterprise plans only)."
        },
        {
            "instruction": "How do I delete my account?",
            "output": "Go to Account Settings > Privacy & Data and click 'Delete My Account'. Warning: This action is permanent and cannot be undone."
        }
    ]
    
    # Format for training
    formatted_data = []
    for item in training_data:
        formatted_data.append({
            "text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        })
    
    return Dataset.from_list(formatted_data)


def main():
    """Main fine-tuning function."""
    
    print("="*80)
    print("UNSLOTH FINE-TUNING DEMO")
    print("="*80)
    print("\nThis demo fine-tunes a 1B model (TinyLlama) for customer support.")
    print("⚠️  Requires GPU with at least 8GB VRAM")
    print("⚠️  This will take several minutes\n")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ No GPU detected! Fine-tuning requires GPU.")
        print("   Consider using Google Colab or a cloud GPU instance.")
        return
    
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Model selection
    model_name = "unsloth/tinyllama-bnb-4bit"  # 1B model, 4-bit quantized
    
    print(f"📦 Loading model: {model_name}")
    print("   This may take a minute...\n")
    
    try:
        # Load model with 4-bit quantization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
        
        print("✅ Model loaded successfully!")
        
        # Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning
        print("\n🔧 Setting up LoRA for efficient fine-tuning...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            use_gradient_checkpointing=True,
            random_state=3407,
        )
        
        print("✅ LoRA configured!")
        
        # Prepare dataset
        print("\n📝 Preparing training dataset...")
        dataset = create_training_dataset()
        print(f"✅ Dataset ready: {len(dataset)} examples")
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=20,  # Very short for demo
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        )
        
        # Setup trainer
        print("\n🚀 Starting fine-tuning...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            max_seq_length=2048,
            dataset_text_field="text",
            args=training_args,
            packing=False,
        )
        
        # Train
        trainer.train()
        
        print("\n✅ Fine-tuning complete!")
        
        # Enable fast inference
        FastLanguageModel.for_inference(model)
        
        # Test the model
        print("\n" + "="*80)
        print("TESTING FINE-TUNED MODEL")
        print("="*80)
        
        test_questions = [
            "How do I reset my password?",
            "What are the pricing plans?",
        ]
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            prompt = f"### Instruction:\n{question}\n\n### Response:\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            print(f"Answer: {response}")
        
        print("\n" + "="*80)
        print("FINE-TUNING COMPLETE!")
        print("="*80)
        print("\n💡 Key Takeaways:")
        print("  • Fine-tuning teaches the model domain-specific patterns")
        print("  • Model 'remembers' training data")
        print("  • No retrieval needed at inference time")
        print("  • But: Hard to update, requires retraining")
        
        # Save model (optional)
        print("\n💾 To save the model:")
        print("   model.save_pretrained('fine_tuned_model')")
        print("   tokenizer.save_pretrained('fine_tuned_model')")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  • Ensure GPU is available")
        print("  • Check CUDA drivers are installed")
        print("  • Verify Unsloth is installed: pip install unsloth")
        raise


if __name__ == "__main__":
    main()

