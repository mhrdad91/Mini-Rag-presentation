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

**Note:** The fine-tuning dependencies (Unsloth) are optional and require:
- **NVIDIA, AMD, or Intel GPU** (NOT Apple Silicon/Mac)
- CUDA drivers installed
- Additional dependencies installed separately

Installation (Linux/Windows with GPU only):
```bash
# Step 1: Install base packages
pip install torch transformers trl datasets bitsandbytes

# Step 2: Install unsloth
pip install "unsloth[colab-new]" --no-deps
pip install unsloth_zoo diffusers torchvision
```

**macOS users:** Unsloth requires NVIDIA/AMD/Intel GPUs and will NOT work on Apple Silicon. The RAG demo works perfectly without GPU on all platforms.

4. **Set up environment variables:**

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_api_key_here
```

Or export it in your shell:

```bash
export OPENAI_API_KEY=your_api_key_here
```

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

5. **Compare RAG vs Fine-Tuning (optional):**
   ```bash
   python code/05_rag_vs_finetuning.py
   ```

6. **Fine-tune with Unsloth (optional, requires GPU):**
   ```bash
   python code/06_finetune_unsloth.py
   ```

### Fine-Tuning Demo (Optional)

To run the fine-tuning comparison demo, you'll need:

1. **GPU with at least 8GB VRAM**
2. **CUDA drivers installed**
3. **Additional dependencies:**
   ```bash
   pip install unsloth torch transformers trl datasets bitsandbytes
   ```

**Note:** Fine-tuning is optional and requires significant computational resources. The RAG demo works without GPU.

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

