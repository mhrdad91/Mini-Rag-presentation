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


def export_using_mlx_fuse(model_name, adapter_path, output_path, gguf_path=None):
    """
    Export using mlx_lm.fuse to merge adapter with base model.
    Can export to HuggingFace format (safetensors) or GGUF format (for LM Studio).
    """
    print("\n" + "=" * 80)
    print("STEP 1: FUSING ADAPTER WITH BASE MODEL")
    print("=" * 80)
    print()
    
    # Use mlx_lm.fuse to merge adapter with base model
    print(f"[INFO] Fusing adapter: {adapter_path}")
    print(f"[INFO] With base model: {model_name}")
    print(f"[INFO] Output: {output_path}")
    if gguf_path:
        print(f"[INFO] GGUF output: {gguf_path}")
    print()
    
    # Build command - export both HuggingFace and GGUF formats
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_name,
        "--adapter-path", adapter_path,
        "--save-path", str(output_path)
    ]
    
    # Add GGUF export if requested
    if gguf_path:
        cmd.extend(["--export-gguf", "--gguf-path", str(gguf_path)])
        print("[INFO] Will export to GGUF format for LM Studio compatibility")
    
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
            if gguf_path:
                print(f"[OK] GGUF model saved to: {gguf_path}")
            return True
        else:
            print(f"\n[ERROR] Fusing failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Error during fusing: {e}")
        return False


def convert_to_gguf_with_llamacpp(hf_model_path, output_gguf_path):
    """
    Convert HuggingFace model to GGUF format using llama.cpp's convert script.
    This is required for LM Studio since it doesn't support safetensors.
    """
    print("\n" + "=" * 80)
    print("STEP 2: CONVERTING TO GGUF FORMAT FOR LM STUDIO")
    print("=" * 80)
    print()
    
    # Convert to Path objects if strings
    output_gguf_path = Path(output_gguf_path)
    hf_model_path = Path(hf_model_path)
    
    # Check if llama.cpp convert script exists
    # Try common locations
    convert_script_paths = [
        Path("llama.cpp/convert-hf-to-gguf.py"),
        Path("../llama.cpp/convert-hf-to-gguf.py"),
        Path.home() / "llama.cpp/convert-hf-to-gguf.py",
    ]
    
    convert_script = None
    for path in convert_script_paths:
        if path.exists():
            convert_script = path
            break
    
    if not convert_script:
        print("[WARNING] llama.cpp convert script not found.")
        print("[INFO] LM Studio requires GGUF format. You need to convert manually:")
        print()
        print("1. Install llama.cpp:")
        print("   git clone https://github.com/ggerganov/llama.cpp.git")
        print("   cd llama.cpp")
        print("   pip install -r requirements.txt")
        print()
        print("2. Convert to GGUF:")
        print(f"   python convert-hf-to-gguf.py {hf_model_path}")
        print(f"   # This will create: {output_gguf_path.parent}/ggml-model-f16.gguf")
        print()
        print("3. Move/rename the file:")
        print(f"   mv {output_gguf_path.parent}/ggml-model-f16.gguf {output_gguf_path}")
        print()
        print("4. Quantize (optional, for smaller file size):")
        print(f"   ./llama-quantize {output_gguf_path} {output_gguf_path.parent}/techcorp-qwen2.5-1.5b-instruct-q4_0.gguf Q4_0")
        print()
        return False
    
    print(f"[INFO] Found llama.cpp convert script: {convert_script}")
    print(f"[INFO] Converting: {hf_model_path}")
    print(f"[INFO] Output: {output_gguf_path}")
    print()
    
    cmd = [
        sys.executable,
        str(convert_script),
        str(hf_model_path),
        "--outfile", str(output_gguf_path)
    ]
    
    print(f"[INFO] Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            cwd=str(convert_script.parent)
        )
        
        if result.returncode == 0:
            if output_gguf_path.exists():
                print(f"\n[OK] GGUF model saved to: {output_gguf_path}")
                return True
            else:
                # Sometimes the output goes to a different location
                possible_outputs = [
                    output_gguf_path.parent / "ggml-model-f16.gguf",
                    output_gguf_path.parent / "ggml-model.gguf",
                ]
                for possible in possible_outputs:
                    if possible.exists():
                        print(f"\n[OK] GGUF model saved to: {possible}")
                        print(f"[INFO] Renaming to: {output_gguf_path}")
                        possible.rename(output_gguf_path)
                        return True
                print(f"\n[WARNING] GGUF conversion completed but output file not found at expected location")
                return False
        else:
            print(f"\n[ERROR] GGUF conversion failed with return code {result.returncode}")
            print("[INFO] You may need to convert manually using the instructions above")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Error during GGUF conversion: {e}")
        print("[INFO] You may need to convert manually using the instructions above")
        return False


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
    gguf_path = output_path / "techcorp-qwen2.5-1.5b-instruct.gguf"
    
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
    
    # Step 1: Export to HuggingFace format (safetensors)
    print("[INFO] Step 1: Fusing adapter with base model (HuggingFace format)...")
    print()
    
    success = export_using_mlx_fuse(model_name, str(adapter_path), str(output_path), gguf_path=None)
    
    if not success:
        print("\n[ERROR] Export failed. Check the error messages above.")
        return
    
    # Step 2: Convert to GGUF for LM Studio
    print()
    print("[INFO] Step 2: Converting to GGUF format for LM Studio...")
    print("[INFO] Note: Qwen2 models don't support direct GGUF export in mlx_lm.fuse")
    print("[INFO] Using llama.cpp conversion instead...")
    print()
    
    gguf_success = convert_to_gguf_with_llamacpp(str(output_path), str(gguf_path))
    
    if not gguf_success:
        print("\n[WARNING] Automatic GGUF conversion failed or llama.cpp not found.")
        print("[INFO] You'll need to convert manually. See instructions below.")
        print()
    
    # Show next steps
    print("\n" + "=" * 80)
    print("EXPORT STATUS")
    print("=" * 80)
    print()
    print(f"[OK] HuggingFace model exported to: {output_path}")
    if gguf_success:
        print(f"[OK] GGUF model exported to: {gguf_path}")
        print()
        print("To use in LM Studio:")
        print("1. Open LM Studio")
        print("2. Go to 'Browse' or 'Local Models'")
        print(f"3. Navigate to: {gguf_path.absolute()}")
        print("4. Select the .gguf file")
        print("5. Load and chat!")
    else:
        print("[WARNING] GGUF conversion not completed automatically")
        print()
        print("LM Studio requires GGUF format. To convert manually:")
        print()
        print("1. Install llama.cpp:")
        print("   git clone https://github.com/ggerganov/llama.cpp.git")
        print("   cd llama.cpp")
        print("   pip install -r requirements.txt")
        print()
        print("2. Convert HuggingFace model to GGUF:")
        print(f"   python convert-hf-to-gguf.py {output_path.absolute()}")
        print()
        print("3. The output will be in the same directory as the convert script")
        print("   Look for: ggml-model-f16.gguf")
        print()
        print("4. Move it to your models directory:")
        print(f"   mv llama.cpp/ggml-model-f16.gguf {gguf_path}")
        print()
    print()
    
    # Optional: Show quantization info
    print("=" * 80)
    print("OPTIONAL: QUANTIZE FOR SMALLER FILE SIZE")
    print("=" * 80)
    print()
    print("The exported GGUF file is in F16 format (large file size).")
    print("To create a smaller quantized version, use llama.cpp:")
    print()
    print("1. Install llama.cpp:")
    print("   git clone https://github.com/ggerganov/llama.cpp.git")
    print("   cd llama.cpp")
    print("   make")
    print()
    print("2. Quantize the model:")
    print(f"   ./llama-quantize {gguf_path} {output_path}/techcorp-qwen2.5-1.5b-instruct-q4_0.gguf Q4_0")
    print()
    print("Quantization options:")
    print("  Q4_0  - Smallest, fastest (recommended for most users)")
    print("  Q4_1  - Slightly larger, better quality")
    print("  Q5_0  - Balanced quality and size")
    print("  Q8_0  - Highest quality, larger file")
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

