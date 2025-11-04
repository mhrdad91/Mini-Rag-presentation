# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (for embeddings and LLM)
- Internet connection for first-time setup

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Note:** Fine-tuning dependencies are optional and platform-specific:

### For Apple Silicon Macs (M1/M2/M3):
Use **MLX** (Apple's optimized framework):
```bash
pip install mlx mlx-lm transformers datasets
```
Then run: `python code/07_finetune_mlx.py`

**Advantages:**
- Native Apple framework, optimized for M-series chips
- Uses Metal Performance Shaders for GPU acceleration
- Supports LoRA/QLoRA fine-tuning
- Can fine-tune 7B models on 16GB Macs

### For NVIDIA/AMD GPUs (Linux/Windows):
Use **Unsloth**:
```bash
# Step 1: Install base packages
pip install torch transformers trl datasets bitsandbytes

# Step 2: Install unsloth
pip install "unsloth[colab-new]" --no-deps
pip install unsloth_zoo diffusers torchvision
```

**Note:** Unsloth requires NVIDIA/AMD/Intel GPUs and will NOT work on Apple Silicon.

The RAG demo works perfectly without GPU on all platforms!

4. **Set up environment variables:**

Create a `.env` file in the root directory (copy from `.env.example`):

**Option 1: Use OpenRouter (Recommended - supports embeddings and LLM):**
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Option 2: Use OpenAI directly:**
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**Note:** The system will automatically use OpenRouter if `OPENROUTER_API_KEY` is set, otherwise it will use OpenAI if `OPENAI_API_KEY` is set.

Get API keys from:
- OpenRouter: https://openrouter.ai/keys
- OpenAI: https://platform.openai.com/api-keys

## Running the Presentation

### Quick Test (Recommended First Step)

**Test everything works:**

```bash
python run_all.py
```

This script will:
- Check all dependencies
- Verify environment setup
- Run all steps in sequence
- Test the complete system
- Report any issues

### Step-by-Step Execution

If you prefer to run steps manually:

1. **Load documents:**
   ```bash
   python code/01_load_documents.py
   ```

2. **Create vector store:**
   ```bash
   python code/02_create_vectorstore.py
   ```

3. **Build RAG system:**
   ```bash
   python code/03_build_rag.py
   ```

4. **Run interactive chatbot:**
   ```bash
   python code/04_chatbot.py
   ```

5. **Run web-based chatbot (Recommended for presentations):**
   ```bash
   streamlit run code/web_chatbot.py
   ```
   
   Or use the launcher:
   ```bash
   python run_web_demo.py
   ```
   
   The web interface will open at `http://localhost:8501`

6. **Compare RAG vs Fine-Tuning (optional):**
   ```bash
   python code/05_rag_vs_finetuning.py
   ```

6. **Fine-tune with Unsloth (optional, requires NVIDIA/AMD GPU):**
   ```bash
   python code/06_finetune_unsloth.py
   ```

7. **Fine-tune with MLX (optional, requires Apple Silicon Mac):**
   ```bash
   python code/08_finetune_mlx_complete.py
   ```
   
   This will:
   - Extract Q&A pairs from your knowledge base
   - Create training data
   - Fine-tune a Qwen2.5 1.5B Instruct model
   - Save your custom model

   See `FINE_TUNING_GUIDE.md` for detailed instructions.

### Fine-Tuning Demo (Optional)

Fine-tuning options depend on your platform:

**Apple Silicon Macs (M1/M2/M3):**
- Use MLX framework (Apple's optimized solution)
- Install: `pip install mlx mlx-lm transformers datasets`
- Run: `python code/07_finetune_mlx.py`
- Can fine-tune 7B models on 16GB Macs with LoRA

**NVIDIA/AMD GPUs (Linux/Windows):**
- Use Unsloth framework
- Requires GPU with at least 8GB VRAM
- CUDA drivers installed
- Install: `pip install unsloth torch transformers trl datasets bitsandbytes`

**Note:** Fine-tuning is optional and requires significant computational resources. The RAG demo works without GPU on all platforms.

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError`
- **Solution:** Make sure virtual environment is activated and dependencies are installed

**Issue:** `OpenAI API key not found`
- **Solution:** Check your `.env` file or environment variables

**Issue:** `No module named 'langchain'`
- **Solution:** Run `pip install -r requirements.txt` again

**Issue:** Vector store not found
- **Solution:** Run `02_create_vectorstore.py` first to create the vector store

## File Structure

```
.
├── presentation.md          # Main presentation file
├── code/
│   ├── 01_load_documents.py
│   ├── 02_create_vectorstore.py
│   ├── 03_build_rag.py
│   └── 04_chatbot.py
├── knowledge_base/         # Sample documents
├── vectorstore/            # Generated vector store (created after step 2)
└── requirements.txt        # Python dependencies
```

## Notes

- The vector store is saved locally in the `vectorstore/` directory
- You can modify the knowledge base documents in `knowledge_base/` and rebuild
- All code files are designed to be run independently and in sequence

