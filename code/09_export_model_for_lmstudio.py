"""
Export Fine-Tuned MLX Model for LM Studio
==========================================

This script exports a fine-tuned MLX model to a format compatible with LM Studio.
LM Studio can load models in HuggingFace format or GGUF format.

Requirements:
- Fine-tuned adapter must exist (from code/08_finetune_mlx_complete.py)
- MLX framework installed
- Optional: llama.cpp for GGUF conversion

Run with:
    python code/09_export_model_for_lmstudio.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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


def export_using_mlx_fuse(model_name, adapter_path, output_path):
    """
    Export using mlx_lm.fuse to merge adapter with base model.
    This creates a complete model that can be converted to HuggingFace format.
    """
    print("\n" + "=" * 80)
    print("STEP 1: FUSING ADAPTER WITH BASE MODEL")
    print("=" * 80)
    print()
    
    # Use mlx_lm.fuse to merge adapter with base model
    print(f"[INFO] Fusing adapter: {adapter_path}")
    print(f"[INFO] With base model: {model_name}")
    print(f"[INFO] Output: {output_path}")
    print()
    
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_name,
        "--adapter-path", adapter_path,
        "--save-path", str(output_path)
    ]
    
    print(f"[INFO] Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n[OK] Model fused and saved to: {output_path}")
            return True
        else:
            print(f"\n[ERROR] Fusing failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Error during fusing: {e}")
        return False


def export_to_gguf(hf_model_path, output_gguf_path):
    """
    Convert HuggingFace model to GGUF format using llama.cpp.
    This is optional - LM Studio can also load HuggingFace models directly.
    """
    print("\n" + "=" * 80)
    print("STEP 2: CONVERTING TO GGUF FORMAT (Optional)")
    print("=" * 80)
    print()
    
    print("[INFO] GGUF format is optional - LM Studio can load HuggingFace models directly.")
    print("[INFO] To convert to GGUF, you need llama.cpp:")
    print()
    print("1. Clone llama.cpp:")
    print("   git clone https://github.com/ggerganov/llama.cpp.git")
    print("   cd llama.cpp")
    print()
    print("2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("3. Convert to GGUF:")
    print(f"   python convert-hf-to-gguf.py {hf_model_path}")
    print()
    print("4. Quantize (optional, for smaller file size):")
    print("   ./llama-quantize <model-f16.gguf> <model-q4_0.gguf> Q4_0")
    print()
    
    return False  # Return False since we're not doing it automatically


def main():
    """Main function to export model for LM Studio"""
    print("=" * 80)
    print("EXPORT FINE-TUNED MODEL FOR LM STUDIO")
    print("=" * 80)
    print()
    
    # Check platform
    if not check_apple_silicon():
        print("[WARNING] This script is optimized for Apple Silicon Macs.")
        print()
    
    # Check MLX
    if not check_mlx_available():
        print("[ERROR] MLX not installed!")
        print("Install with: pip install mlx mlx-lm")
        return
    
    print("[OK] MLX framework detected")
    print()
    
    # Model configuration
    model_name = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    adapter_path = Path("adapters/techcorp-support")
    output_path = Path("models/techcorp-qwen2.5-1.5b-instruct")
    
    # Check if adapter exists
    if not adapter_path.exists():
        print(f"[ERROR] Fine-tuned adapter not found at: {adapter_path}")
        print()
        print("Please run fine-tuning first:")
        print("  python code/08_finetune_mlx_complete.py")
        return
    
    print(f"[OK] Found adapter at: {adapter_path}")
    print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export using mlx_lm.fuse
    success = export_using_mlx_fuse(model_name, str(adapter_path), str(output_path))
    
    if not success:
        print("\n[ERROR] Export failed. Check the error messages above.")
        return
    
    # Show next steps
    print("\n" + "=" * 80)
    print("EXPORT COMPLETE!")
    print("=" * 80)
    print()
    print(f"Model exported to: {output_path}")
    print()
    print("To use in LM Studio:")
    print("1. Open LM Studio")
    print("2. Go to 'Browse' or 'Local Models'")
    print(f"3. Navigate to: {output_path.absolute()}")
    print("4. Select the model folder")
    print("5. Load and chat!")
    print()
    print("Note: LM Studio can load HuggingFace format models directly.")
    print("If you want GGUF format (smaller file size), follow the instructions above.")
    print()
    
    # Optional GGUF conversion info
    export_to_gguf(str(output_path), None)


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

