"""
Fine-Tuning Demo with MLX (Apple Silicon)
==========================================

This script demonstrates fine-tuning a small model using MLX framework,
which is optimized for Apple Silicon (M1/M2/M3) chips.

MLX is Apple's native framework that leverages Metal Performance Shaders
for efficient training on Mac hardware.

Requirements:
- Apple Silicon Mac (M1/M2/M3)
- MLX installed: pip install mlx mlx-lm transformers datasets

Note: This is designed for Apple Silicon. For NVIDIA/AMD GPUs, use Unsloth instead.
"""

import os
import json
import platform
from pathlib import Path
from dotenv import load_dotenv

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
        from mlx_lm import train, load, utils
        return True
    except ImportError:
        return False


def create_training_dataset():
    """
    Create training dataset for customer support fine-tuning.
    Format compatible with MLX.
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
            "instruction": "What is your refund policy?",
            "output": "We offer a 30-day money-back guarantee. Refunds are processed within 5-7 business days. Contact support@techcorp.com for assistance."
        },
        {
            "instruction": "How do I add team members?",
            "output": "Go to Settings > Team Members, click 'Add Member', enter their email address, and select their role. They'll receive an invitation email."
        },
        {
            "instruction": "What is data privacy?",
            "output": "We take data privacy seriously. Your data is encrypted in transit and at rest. We comply with GDPR and SOC 2 Type II standards. We never sell your data."
        },
        {
            "instruction": "How do I export my data?",
            "output": "Go to Settings > Data Management > Export Data. You can export all your data in JSON or CSV format. Large exports may take a few minutes."
        },
        {
            "instruction": "What browsers are supported?",
            "output": "We support Chrome, Firefox, Safari, and Edge (latest versions). For best experience, use Chrome or Firefox."
        },
    ]
    
    # Convert to MLX format (simple text format)
    mlx_data = []
    for item in training_data:
        # Format: instruction + response
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


def main():
    print("=" * 80)
    print("MLX FINE-TUNING DEMO FOR APPLE SILICON")
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
        print("MLX is Apple's framework optimized for Apple Silicon.")
        print("It uses Metal Performance Shaders for GPU acceleration.")
        return
    
    print("[OK] MLX framework detected")
    print()
    
    # Create training dataset
    print("Creating training dataset...")
    training_data = create_training_dataset()
    data_path = save_training_data(training_data)
    
    print()
    print("=" * 80)
    print("MLX FINE-TUNING SETUP")
    print("=" * 80)
    print()
    print("MLX Setup Instructions:")
    print("-" * 80)
    print()
    print("1. Install MLX:")
    print("   pip install mlx mlx-lm transformers datasets")
    print()
    print("2. Prepare your data:")
    print(f"   Data saved to: {data_path}")
    print()
    print("3. Fine-tune a model (example with Qwen2.5 1.5B):")
    print("   python -m mlx_lm lora --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\")
    print("                --data {data_path} \\")
    print("                --adapter-path adapters/qwen-customer-support \\")
    print("                --iters 100 \\")
    print("                --learning-rate 1e-4 \\")
    print("                --batch-size 2 \\")
    print("                --val-batches 10")
    print()
    print("4. Generate with fine-tuned model:")
    print("   python -m mlx_lm generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\")
    print("                    --adapter-path adapters/qwen-customer-support \\")
    print("                    --prompt 'How do I reset my password?'")
    print()
    print("=" * 80)
    print("KEY DIFFERENCES: MLX vs Unsloth")
    print("=" * 80)
    print()
    print("MLX (Apple Silicon):")
    print("  [+] Native Apple framework, optimized for M1/M2/M3")
    print("  [+] Uses Metal Performance Shaders (GPU acceleration)")
    print("  [+] Works seamlessly on macOS")
    print("  [+] Supports LoRA and QLoRA fine-tuning")
    print("  [+] No CUDA/NVIDIA dependencies needed")
    print("  [-] Only works on Apple Silicon Macs")
    print()
    print("Unsloth (NVIDIA/AMD GPUs):")
    print("  [+] Optimized for NVIDIA GPUs")
    print("  [+] Supports larger models")
    print("  [+] Very fast on CUDA")
    print("  [-] Requires NVIDIA/AMD/Intel GPU")
    print("  [-] Does NOT work on Apple Silicon")
    print()
    print("=" * 80)
    print("ADVANTAGES OF MLX ON APPLE SILICON")
    print("=" * 80)
    print()
    print("1. Native Performance:")
    print("   - MLX is optimized specifically for Apple Silicon")
    print("   - Leverages unified memory architecture")
    print("   - Uses Metal for GPU acceleration")
    print()
    print("2. Easy Setup:")
    print("   - No CUDA drivers needed")
    print("   - Works out of the box on macOS")
    print("   - Simple pip install")
    print()
    print("3. Memory Efficient:")
    print("   - Supports LoRA (Low-Rank Adaptation)")
    print("   - Supports QLoRA (Quantized LoRA)")
    print("   - Can fine-tune 7B models on 16GB Macs")
    print()
    print("4. Production Ready:")
    print("   - Developed and maintained by Apple")
    print("   - Active community support")
    print("   - Regular updates and optimizations")
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("To actually run fine-tuning, use the MLX command-line tools:")
    print()
    print("  mlx_lm.lora --help  # See all options")
    print()
    print("For more information, visit:")
    print("  https://github.com/ml-explore/mlx-examples")
    print()


if __name__ == "__main__":
    main()

