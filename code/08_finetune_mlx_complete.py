"""
Complete Fine-Tuning Demo with MLX (Apple Silicon)
==================================================

This script demonstrates fine-tuning a small model using MLX framework
for customer support tasks.

The script will:
1. Extract Q&A pairs from knowledge base
2. Format data for MLX fine-tuning
3. Fine-tune a small model using LoRA
4. Test the fine-tuned model
5. Compare with base model

Requirements:
- Apple Silicon Mac (M1/M2/M3)
- MLX installed: pip install mlx mlx-lm transformers datasets

Run with:
    python code/08_finetune_mlx_complete.py
"""

import os
import json
import platform
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

load_dotenv()

# Check if running on Apple Silicon
def check_apple_silicon():
    """Check if running on Apple Silicon"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def check_mlx_available():
    """Check if MLX is available"""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        return True
    except ImportError:
        return False


def extract_qa_from_knowledge_base():
    """
    Extract Q&A pairs from knowledge base documents.
    This creates training data from the actual knowledge base.
    """
    knowledge_base_path = Path("knowledge_base")
    
    if not knowledge_base_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {knowledge_base_path}")
    
    # Extract Q&A pairs from markdown files
    qa_pairs = []
    
    # Read FAQ file
    faq_path = knowledge_base_path / "faq.md"
    if faq_path.exists():
        with open(faq_path, "r") as f:
            content = f.read()
            
        # Extract Q&A pairs from FAQ format
        # Look for "###" headers (questions) followed by content (answers)
        lines = content.split("\n")
        current_question = None
        current_answer = []
        
        for line in lines:
            if line.startswith("###"):
                if current_question and current_answer:
                    answer = "\n".join(current_answer).strip()
                    if answer:
                        qa_pairs.append({
                            "instruction": current_question,
                            "output": answer
                        })
                current_question = line.replace("###", "").strip()
                current_answer = []
            elif current_question and line.strip():
                current_answer.append(line)
        
        # Add last Q&A pair
        if current_question and current_answer:
            answer = "\n".join(current_answer).strip()
            if answer:
                qa_pairs.append({
                    "instruction": current_question,
                    "output": answer
                })
    
    # Add specific Q&A pairs from knowledge base
    training_data = [
        {
            "instruction": "How do I reset my password?",
            "output": "To reset your password, go to the login page, click 'Forgot Password,' enter your email, and follow the instructions in the email you receive. Note: Password reset links expire after 24 hours for security."
        },
        {
            "instruction": "What are the pricing plans?",
            "output": "We offer Free ($0/month), Starter ($10/user/month), Professional ($25/user/month), and Enterprise (custom pricing). The Free plan includes 3 projects and 5 team members."
        },
        {
            "instruction": "How do I cancel my subscription?",
            "output": "Go to Account Settings > Billing and click 'Cancel Subscription.' Your account remains active until the end of your billing period."
        },
        {
            "instruction": "What is your refund policy?",
            "output": "Monthly plans: No refunds for partial months. Annual plans: Refund available for unused months (prorated). Refunds processed within 5-7 business days."
        },
        {
            "instruction": "How do I add team members?",
            "output": "Navigate to 'Team' in the sidebar, click 'Invite Members,' enter email addresses, select their role, and click 'Send Invitations.'"
        },
        {
            "instruction": "What is data privacy?",
            "output": "We take data privacy seriously. Your data is encrypted in transit and at rest. We comply with GDPR and SOC 2 Type II standards. We never sell your data."
        },
        {
            "instruction": "How do I export my data?",
            "output": "Go to Account Settings > Privacy & Data and click 'Request Data Export.' You can export in CSV, JSON, or Excel format."
        },
        {
            "instruction": "What browsers are supported?",
            "output": "We support Chrome, Firefox, Safari, and Edge (latest versions). For best experience, use Chrome or Firefox."
        },
        {
            "instruction": "How many team members can I invite?",
            "output": "Free plan: 5 members, Starter: 25 members, Professional and Enterprise: Unlimited members."
        },
        {
            "instruction": "Can I export my data?",
            "output": "Yes, go to Account Settings > Privacy & Data and click 'Request Data Export.' You can export in CSV, JSON, or Excel format."
        },
        {
            "instruction": "Is there a mobile app?",
            "output": "Yes, we have iOS and Android apps available in the App Store and Google Play Store."
        },
        {
            "instruction": "How do I change my email address?",
            "output": "Go to Account Settings > Profile and update your email. You'll need to verify the new email address."
        },
        {
            "instruction": "What happens if I cancel?",
            "output": "Your account remains active until the end of your billing period. After that, you'll be moved to the Free plan or your account will be deactivated. Data is retained for 30 days."
        },
        {
            "instruction": "Do you offer refunds?",
            "output": "Monthly plans: No refunds for partial months. Annual plans: Refund available for unused months (prorated). Refunds processed within 5-7 business days."
        },
        {
            "instruction": "What payment methods do you accept?",
            "output": "We accept credit cards (Visa, MasterCard, American Express), debit cards, PayPal, and bank transfers (Enterprise plans only)."
        },
    ]
    
    # Combine extracted and manual Q&A pairs
    all_qa = training_data + qa_pairs
    
    # Remove duplicates
    seen = set()
    unique_qa = []
    for qa in all_qa:
        key = qa["instruction"].lower()
        if key not in seen:
            seen.add(key)
            unique_qa.append(qa)
    
    print(f"[OK] Extracted {len(unique_qa)} unique Q&A pairs")
    return unique_qa


def format_for_mlx(training_data):
    """
    Format training data for MLX fine-tuning.
    MLX expects JSONL format with 'text' field.
    """
    mlx_data = []
    for item in training_data:
        # Format as instruction-response pairs
        text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        mlx_data.append({"text": text})
    
    return mlx_data


def save_training_data(data, output_file="training_data.jsonl"):
    """Save training data in JSONL format for MLX"""
    output_path = Path("data") / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"[OK] Training data saved to {output_path}")
    return output_path


def check_mlx_lm_command():
    """Check if mlx_lm command is available"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mlx_lm", "lora", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def fine_tune_with_mlx(data_path, model_name="mlx-community/Qwen2.5-1.5B-Instruct-4bit", 
                       adapter_path="adapters/techcorp-support", iters=100):
    """
    Fine-tune model using MLX command-line tool.
    
    Args:
        data_path: Path to training data JSONL file
        model_name: Base model name
        adapter_path: Where to save the adapter
        iters: Number of training iterations
    """
    print("\n" + "=" * 80)
    print("STARTING MLX FINE-TUNING")
    print("=" * 80)
    print()
    print(f"Model: {model_name}")
    print(f"Training data: {data_path}")
    print(f"Adapter path: {adapter_path}")
    print(f"Iterations: {iters}")
    print()
    
    # Create adapter directory
    adapter_dir = Path(adapter_path).parent
    adapter_dir.mkdir(parents=True, exist_ok=True)
    
    # Build mlx_lm lora command (updated syntax)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_name,
        "--data", str(data_path),
        "--adapter-path", adapter_path,
        "--iters", str(iters),
        "--learning-rate", "1e-4",
        "--batch-size", "2",
        "--val-batches", "5",
        "--train"
    ]
    
    print("[INFO] Running MLX fine-tuning command...")
    print(f"Command: {' '.join(cmd)}")
    print()
    print("[INFO] This may take a few minutes...")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print("\n[OK] Fine-tuning completed successfully!")
            return True
        else:
            print(f"\n[ERROR] Fine-tuning failed with return code {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n[WARNING] Fine-tuning interrupted by user")
        return False
    except Exception as e:
        print(f"\n[ERROR] Error during fine-tuning: {e}")
        return False


def test_fine_tuned_model(model_name, adapter_path, test_prompts):
    """
    Test the fine-tuned model with sample prompts.
    
    Args:
        model_name: Base model name
        adapter_path: Path to fine-tuned adapter
        test_prompts: List of test prompts
    """
    print("\n" + "=" * 80)
    print("TESTING FINE-TUNED MODEL")
    print("=" * 80)
    print()
    
    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        print(f"[ERROR] Adapter not found at {adapter_path}")
        print("[INFO] Run fine-tuning first!")
        return
    
    for prompt in test_prompts:
        print(f"Question: {prompt}")
        print("-" * 80)
        
        cmd = [
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", model_name,
            "--adapter-path", adapter_path,
            "--prompt", prompt,
            "--max-tokens", "200",
            "--temp", "0.3"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Extract the generated text (skip the prompt)
                output = result.stdout
                if "Response:" in output or "### Response:" in output:
                    # Try to extract just the response
                    lines = output.split("\n")
                    in_response = False
                    response_lines = []
                    for line in lines:
                        if "Response:" in line or "### Response:" in line:
                            in_response = True
                            response_lines.append(line.split("Response:")[-1].strip())
                        elif in_response and line.strip():
                            response_lines.append(line.strip())
                    
                    if response_lines:
                        print("\n".join(response_lines))
                    else:
                        print(output)
                else:
                    print(output)
            else:
                print(f"[ERROR] Generation failed: {result.stderr}")
            
            print()
            
        except Exception as e:
            print(f"[ERROR] Error testing model: {e}")
            print()


def main():
    """Main function to run complete fine-tuning process."""
    print("=" * 80)
    print("MLX FINE-TUNING - COMPLETE DEMO")
    print("=" * 80)
    print()
    
    # Check platform
    if not check_apple_silicon():
        print("[WARNING] This script is optimized for Apple Silicon Macs.")
        print("[WARNING] It may work on Intel Macs but performance will be limited.")
        print()
    
    # Check MLX availability
    if not check_mlx_available():
        print("[ERROR] MLX not installed!")
        print()
        print("Install MLX with:")
        print("  pip install mlx mlx-lm transformers datasets")
        print()
        return
    
    print("[OK] MLX framework detected")
    
    # Check mlx_lm command
    if not check_mlx_lm_command():
        print("[WARNING] mlx_lm command-line tool not available")
        print("[INFO] Trying to install mlx-lm...")
        print("  Run: pip install mlx-lm")
        print()
    
    # Extract training data from knowledge base
    print("\n" + "=" * 80)
    print("STEP 1: PREPARING TRAINING DATA")
    print("=" * 80)
    print()
    
    try:
        qa_pairs = extract_qa_from_knowledge_base()
        mlx_data = format_for_mlx(qa_pairs)
        data_path = save_training_data(mlx_data)
        
        print(f"\n[OK] Prepared {len(mlx_data)} training examples")
        print(f"[OK] Data saved to: {data_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to prepare training data: {e}")
        return
    
    # Fine-tune model
    print("\n" + "=" * 80)
    print("STEP 2: FINE-TUNING MODEL")
    print("=" * 80)
    print()
    
    model_name = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    adapter_path = "adapters/techcorp-support"
    
    print("[INFO] Using model: Qwen2.5 1.5B Instruct (4-bit quantized)")
    print("[INFO] This will take approximately 10-20 minutes depending on your Mac")
    print()
    
    # Check if running in non-interactive mode (for testing)
    if not sys.stdin.isatty():
        print("[INFO] Non-interactive mode detected. Skipping fine-tuning.")
        print("[INFO] Run manually with the command below:")
        print()
        print("Manual command:")
        print(f"  python -m mlx_lm lora --model {model_name} \\")
        print(f"    --data {data_path} \\")
        print(f"    --adapter-path {adapter_path} \\")
        print(f"    --iters 100 --train")
        return
    
    user_input = input("Proceed with fine-tuning? (y/n): ").strip().lower()
    if user_input != 'y':
        print("[INFO] Fine-tuning skipped. You can run it manually later.")
        print()
        print("Manual command:")
        print(f"  python -m mlx_lm lora --model {model_name} \\")
        print(f"    --data {data_path} \\")
        print(f"    --adapter-path {adapter_path} \\")
        print(f"    --iters 100 --train")
        return
    
    success = fine_tune_with_mlx(data_path, model_name, adapter_path, iters=100)
    
    if not success:
        print("\n[ERROR] Fine-tuning failed. Check the error messages above.")
        return
    
    # Test fine-tuned model
    print("\n" + "=" * 80)
    print("STEP 3: TESTING FINE-TUNED MODEL")
    print("=" * 80)
    print()
    
    test_prompts = [
        "How do I reset my password?",
        "What are the pricing plans?",
        "How do I cancel my subscription?",
    ]
    
    test_fine_tuned_model(model_name, adapter_path, test_prompts)
    
    # Summary
    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"1. Use the fine-tuned model: python -m mlx_lm generate \\")
    print(f"     --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\")
    print(f"     --adapter-path {adapter_path} \\")
    print(f"     --prompt 'Your question here'")
    print()
    print("2. Compare with RAG:")
    print("   python code/05_rag_vs_finetuning.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

