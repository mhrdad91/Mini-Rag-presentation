#!/usr/bin/env python3
"""
Quick System Check (No API Key Required)
"""
import sys
from pathlib import Path

print('='*80)
print('SYSTEM CHECK RESULTS')
print('='*80)
print()

# Check dependencies
deps = ['langchain', 'langchain_openai', 'langchain_community', 'faiss', 'dotenv']
missing = []
for dep in deps:
    try:
        if dep == 'faiss':
            __import__('faiss')
        elif dep == 'dotenv':
            __import__('dotenv')
        else:
            __import__(dep)
        print(f'[OK] {dep}')
    except ImportError:
        missing.append(dep)
        print(f'[ERROR] {dep} missing')

print()

# Check knowledge base
kb_path = Path('knowledge_base')
if kb_path.exists():
    md_files = list(kb_path.glob('*.md'))
    print(f'[OK] Knowledge base found: {len(md_files)} files')
    for f in md_files[:5]:
        print(f'  • {f.name}')
else:
    print('[ERROR] Knowledge base not found')

print()

# Check code files
code_files = [
    'code/01_load_documents.py',
    'code/02_create_vectorstore.py',
    'code/03_build_rag.py',
    'code/04_chatbot.py',
    'code/05_rag_vs_finetuning.py',
    'code/06_finetune_unsloth.py',
    'code/07_finetune_mlx.py',
]

print('[OK] Code files:')
for f in code_files:
    if Path(f).exists():
        print(f'  • {f}')
    else:
        print(f'  [ERROR] {f} missing')

print()

# Check optional dependencies
print('Optional dependencies:')
try:
    import mlx.core as mx
    print(f'  [OK] mlx installed (Apple Silicon support)')
except ImportError:
    print(f'  [-] mlx not installed (optional for Apple Silicon)')

try:
    import torch
    print(f'  [OK] torch installed')
except ImportError:
    print(f'  [-] torch not installed (optional)')

try:
    import transformers
    print(f'  [OK] transformers installed')
except ImportError:
    print(f'  [-] transformers not installed (optional)')

# Unsloth check (will fail on Apple Silicon)
try:
    import unsloth
    print(f'  [OK] unsloth installed (but requires NVIDIA/AMD GPU)')
except NotImplementedError:
    print(f'  [INFO] unsloth installed (but requires NVIDIA/AMD GPU - not available on Apple Silicon)')
except ImportError:
    print(f'  [-] unsloth not installed (optional, requires NVIDIA/AMD GPU)')

print()

# Check vectorstore
vs_path = Path('vectorstore')
if vs_path.exists():
    files = list(vs_path.iterdir())
    print(f'[OK] Vector store exists ({len(files)} files)')
else:
    print('[INFO] Vector store not created yet (will be created by step 2)')

print()
print('='*80)
print('SUMMARY')
print('='*80)
if not missing:
    print('[SUCCESS] All core dependencies are installed!')
    print()
    print('To run the full RAG system:')
    print('1. Create a .env file with: OPENAI_API_KEY=your_key_here')
    print('2. Then run: python run_all.py')
    print()
    print('For Apple Silicon fine-tuning:')
    print('  pip install mlx mlx-lm')
    print('  python code/07_finetune_mlx.py')
else:
    print(f'[ERROR] Missing dependencies: {", ".join(missing)}')
    print('Install with: pip install -r requirements.txt')
print('='*80)

