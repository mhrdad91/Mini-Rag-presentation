# Fine-Tuning Your Own Model with MLX

## Quick Start

### Step 1: Prepare Training Data

The script automatically extracts Q&A pairs from your knowledge base:

```bash
python code/08_finetune_mlx_complete.py
```

This will:
- Extract Q&A pairs from `knowledge_base/` files
- Format them for MLX fine-tuning
- Save to `data/training_data.jsonl`

### Step 2: Fine-Tune the Model

Run the fine-tuning command:

```bash
python -m mlx_lm lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --data data/training_data.jsonl \
  --adapter-path adapters/techcorp-support \
  --iters 100 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --train
```

**What happens:**
- Downloads Qwen2.5 1.5B Instruct model (first time only)
- Fine-tunes using LoRA (Low-Rank Adaptation)
- Saves adapter to `adapters/techcorp-support/`
- Takes ~10-20 minutes on Apple Silicon Macs

### Step 3: Test Your Fine-Tuned Model

```bash
python -m mlx_lm generate \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path adapters/techcorp-support \
  --prompt "How do I reset my password?" \
  --max-tokens 200
```

### Step 4: Compare Fine-Tuned vs RAG

```bash
python code/05_rag_vs_finetuning.py
```

## What You're Creating

### Fine-Tuned Model (Your Own Model)
- ✅ **Learned** from your training data
- ✅ **No retrieval** needed at inference
- ✅ **Domain-specific** knowledge baked in
- ⚠️ Requires training time
- ⚠️ Hard to update (must retrain)

### RAG System (Current System)
- ✅ **No training** needed
- ✅ **Easy to update** (just change knowledge base)
- ✅ **Always current** (real-time retrieval)
- ⚠️ Requires retrieval step
- ⚠️ Needs vector database

## Training Data

Your training data includes:
- 31 Q&A pairs extracted from knowledge base
- Questions about password reset, pricing, refunds, etc.
- Both standard and fun answers included!

## Model Options

### Qwen2.5 1.5B Instruct (Recommended for Demo)
- **Fast** training (~10-20 minutes)
- **Small** size (~900MB)
- **Good** for presentations
- **Better** quality than smaller models
- **Model:** `mlx-community/Qwen2.5-1.5B-Instruct-4bit`

### Larger Models (If You Have Time)
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- `mlx-community/Llama-3.2-3B-Instruct-v0.1`
- `mlx-community/Qwen2.5-3B-Instruct-4bit`

**Note:** Larger models take longer to train but produce better results.

## Troubleshooting

**mlx_lm command not found:**
```bash
pip install mlx-lm
```

**Out of memory:**
- Use smaller batch size: `--batch-size 1`
- Use smaller model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`

**Training too slow:**
- Reduce iterations: `--iters 50`
- Use smaller model
- Close other applications

## Next Steps After Fine-Tuning

1. **Test your model** with various questions
2. **Compare** with RAG to see differences
3. **Integrate** into your chatbot
4. **Deploy** for production use

## Key Concepts

**LoRA (Low-Rank Adaptation):**
- Only trains small adapter weights
- Keeps base model unchanged
- Much faster and memory-efficient
- Adapter is only ~10-50MB

**Adapter vs Full Model:**
- Adapter: Small file that modifies base model
- Full Model: Complete retrained model (much larger)
- LoRA uses adapters (recommended)

**Fine-Tuning vs RAG:**
- Fine-tuning: Model learns knowledge
- RAG: Model retrieves knowledge
- Both have their place!

