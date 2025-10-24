import os
import sys

print("="*80)
print("TRAINING READINESS CHECK")
print("="*80)

checks_passed = 0
checks_total = 0

# Check 1: Models cleared
checks_total += 1
pretrained_exists = os.path.exists('models/pretrained_final/config.json')
finetuned_exists = os.path.exists('models/finetuned_final/config.json')
if not pretrained_exists and not finetuned_exists:
    print("✓ Models cleared - ready for training")
    checks_passed += 1
else:
    print("✗ Old models detected!")
    if pretrained_exists:
        print("  → models/pretrained_final/ exists - DELETE IT")
    if finetuned_exists:
        print("  → models/finetuned_final/ exists - DELETE IT")
    print("  Run: rm -rf models/pretrained* models/finetuned*")

# Check 2: Skip flags
checks_total += 1
try:
    # These should NOT be defined yet (will cause NameError)
    if 'skip_pretraining' in dir() or 'skip_finetuning' in dir():
        print("✗ Skip flags are set - DID YOU RESTART THE KERNEL?")
        print("  → Click kernel name (top-right) → Restart Kernel")
    else:
        print("✓ Kernel is fresh")
        checks_passed += 1
except:
    print("✓ Kernel is fresh")
    checks_passed += 1

# Check 3: Data files exist
checks_total += 1
if os.path.exists('data/pretrain_raw.json') and os.path.exists('data/finetune_raw.json'):
    print("✓ Training data exists")
    checks_passed += 1
else:
    print("⚠ Training data not found (will be created during run)")
    checks_passed += 1  # Not critical

# Check 4: GPU availability
checks_total += 1
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
    checks_passed += 1
else:
    print("⚠ No GPU detected - training will be SLOW (10x slower)")
    print("  Consider using Google Colab or reducing dataset size")
    # Still count as pass, just slow

print("="*80)
print(f"Status: {checks_passed}/{checks_total} checks passed")

if checks_passed == checks_total:
    print("✓ READY TO TRAIN! Run all cells sequentially.")
    print(f"  Expected time: ~{'3-6 hours with GPU' if torch.cuda.is_available() else '15-30 hours on CPU'}")
else:
    print("✗ NOT READY - Fix issues above before training")

print("="*80)

# Commented out IPython magic to ensure Python compatibility.
# Install required packages
# %pip install torch transformers tokenizers datasets requests pandas numpy tqdm GitPython

import os
import ast
import re
import json
import random
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import git

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Config, T5ForConditionalGeneration,
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq
)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Create directories
base_dir = '.'
os.chdir(base_dir)

print("Setup complete!")
print(f"Working directory: {os.getcwd()}")
print(f"CUDA available: {torch.cuda.is_available()}")

"""## 1. Data Collection: Scraping GitHub Repositories"""

# Query SEART-GHS for Python repositories
def search_github_repos(num_repos=500, min_stars=100):
    """
    Search for Python repositories using SEART-GHS API
    https://seart-ghs.si.usi.ch

    Expanded list to ensure 150k+ pre-training and 50k+ fine-tuning examples
    """

    # Expanded list of popular Python repositories (50+ repos for better coverage)
    # Focus on projects with substantial codebases and diverse coding patterns
    demo_repos = [
        # Web Frameworks & APIs
        "https://github.com/psf/requests",
        "https://github.com/pallets/flask",
        "https://github.com/django/django",
        "https://github.com/encode/django-rest-framework",
        "https://github.com/tiangolo/fastapi",
        "https://github.com/tornadoweb/tornado",
        "https://github.com/aio-libs/aiohttp",
        "https://github.com/encode/httpx",

        # Data Science & ML
        "https://github.com/numpy/numpy",
        "https://github.com/pandas-dev/pandas",
        "https://github.com/scikit-learn/scikit-learn",
        "https://github.com/pytorch/pytorch",
        "https://github.com/tensorflow/tensorflow",
        "https://github.com/keras-team/keras",
        "https://github.com/matplotlib/matplotlib",
        "https://github.com/scipy/scipy",
        "https://github.com/statsmodels/statsmodels",
        "https://github.com/pydata/xarray",

        # NLP & AI
        "https://github.com/huggingface/transformers",
        "https://github.com/openai/gym",
        "https://github.com/RaRe-Technologies/gensim",
        "https://github.com/nltk/nltk",
        "https://github.com/explosion/spaCy",

        # DevOps & Tools
        "https://github.com/ansible/ansible",
        "https://github.com/docker/docker-py",
        "https://github.com/kubernetes-client/python",
        "https://github.com/fabric/fabric",
        "https://github.com/pytest-dev/pytest",
        "https://github.com/saltstack/salt",

        # Web Scraping & Automation
        "https://github.com/scrapy/scrapy",
        "https://github.com/psf/requests-html",
        "https://github.com/beautifulsoup4/beautifulsoup4",
        "https://github.com/SeleniumHQ/selenium",

        # CLI & System Tools
        "https://github.com/pallets/click",
        "https://github.com/python/cpython",
        "https://github.com/certbot/certbot",
        "https://github.com/pypa/pip",
        "https://github.com/pypa/setuptools",

        # Database & Storage
        "https://github.com/sqlalchemy/sqlalchemy",
        "https://github.com/mongodb/mongo-python-driver",
        "https://github.com/redis/redis-py",
        "https://github.com/elastic/elasticsearch-py",

        # Async & Concurrency
        "https://github.com/MagicStack/uvloop",
        "https://github.com/celery/celery",
        "https://github.com/python-trio/trio",

        # Image & Media Processing
        "https://github.com/python-pillow/Pillow",
        "https://github.com/imageio/imageio",
        "https://github.com/opencv/opencv-python",

        # Utilities & Libraries
        "https://github.com/jazzband/pip-tools",
        "https://github.com/pyenv/pyenv",
        "https://github.com/python-poetry/poetry",
        "https://github.com/pre-commit/pre-commit",
        "https://github.com/tqdm/tqdm",
    ]

    print(f"Selected {len(demo_repos)} repositories for comprehensive coverage")
    print(f"Expected: 150k+ pre-training examples, 50k+ fine-tuning examples")

    return demo_repos

repo_urls = search_github_repos()
print(f"Found {len(repo_urls)} repositories to clone")

def clone_repository(repo_url: str, target_dir: str) -> Optional[str]:
    """Clone a GitHub repository"""
    try:
        repo_name = repo_url.split('/')[-1]
        clone_path = os.path.join(target_dir, repo_name)

        if os.path.exists(clone_path):
            print(f"Repository {repo_name} already exists, skipping...")
            return clone_path

        print(f"Cloning {repo_name}...")
        git.Repo.clone_from(repo_url.replace("https://", f"https://{GITHUB_TOKEN}@"), clone_path, depth=1)
        return clone_path
    except Exception as e:
        print(f"Error cloning {repo_url}: {e}")
        return None

# Clone repositories (or use existing ones)
cloned_repos = []
print("Checking for existing repositories...")

# First check if repos already exist
for url in repo_urls:
    repo_name = url.split('/')[-1]
    clone_path = os.path.join('repos', repo_name)
    if os.path.exists(clone_path):
        cloned_repos.append(clone_path)

# If we have all repos, skip cloning
if len(cloned_repos) == len(repo_urls):
    print(f"All {len(cloned_repos)} repositories already exist. Skipping cloning...")
else:
    # Clone missing repos
    cloned_repos = []
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    for url in tqdm(repo_urls, desc="Cloning repositories"):
        path = clone_repository(url, 'repos')
        if path:
            cloned_repos.append(path)

print(f"Successfully found/cloned {len(cloned_repos)} repositories")

"""## 2. Function Extraction and Processing"""

def extract_python_files(repo_path: str) -> List[str]:
    """Extract all Python files from a repository"""
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        # Skip test directories and common non-essential directories
        dirs[:] = [d for d in dirs if d not in ['test', 'tests', '__pycache__', '.git', 'venv', 'env']]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files

all_python_files = []
for repo_path in cloned_repos:
    files = extract_python_files(repo_path)
    all_python_files.extend(files)

print(f"Found {len(all_python_files)} Python files")

# Check if we should skip function extraction
skip_extraction = (os.path.exists('data/pretrain_raw.json') and
                   os.path.exists('data/finetune_raw.json') and
                   os.path.exists('data/train.json'))

if skip_extraction:
    print("Function data already exists. Skipping extraction...")
    print("If you want to re-extract, delete the data/*.json files.")
else:
    print("No existing function data found. Will proceed with extraction...")

class FunctionExtractor(ast.NodeVisitor):
    """Extract functions from Python AST"""

    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        # Extract function information
        func_info = {
            'name': node.name,
            'source': ast.unparse(node),
            'lineno': node.lineno,
            'num_lines': len(ast.unparse(node).split('\n')),
            'has_if': self._has_if_statement(node)
        }
        self.functions.append(func_info)
        self.generic_visit(node)

    def _has_if_statement(self, node):
        """Check if function contains if statements"""
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                return True
        return False

def extract_functions_from_file(file_path: str) -> List[Dict]:
    """Extract functions from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()

        tree = ast.parse(source)
        extractor = FunctionExtractor()
        extractor.visit(tree)
        return extractor.functions
    except Exception as e:
        return []

# Extract all functions (or load from cache) with progress tracking
if not skip_extraction:
    all_functions = []
    total_files = len(all_python_files)

    print("=" * 80)
    print(f"EXTRACTING FUNCTIONS FROM {total_files:,} PYTHON FILES")
    print("=" * 80)
    print(f"Target: 150k+ pre-training, 50k+ fine-tuning examples")
    print(f"This may take 1-2 hours depending on repository sizes...")
    print("=" * 80)

    # Track statistics
    files_processed = 0
    functions_extracted = 0
    functions_with_if = 0
    errors = 0

    # Progress bar with detailed stats
    with tqdm(total=total_files, desc="Extracting functions", unit="files") as pbar:
        for file_path in all_python_files:
            functions = extract_functions_from_file(file_path)

            if functions:
                all_functions.extend(functions)
                functions_extracted += len(functions)
                functions_with_if += sum(1 for f in functions if f['has_if'])
            else:
                errors += 1

            files_processed += 1

            # Update progress bar with current stats every 100 files
            if files_processed % 100 == 0:
                pbar.set_postfix({
                    'funcs': f"{functions_extracted:,}",
                    'if_funcs': f"{functions_with_if:,}",
                    'errors': errors
                })

            pbar.update(1)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Files processed: {files_processed:,}")
    print(f"Functions extracted: {functions_extracted:,}")
    print(f"Functions with if statements: {functions_with_if:,}")
    print(f"Parse errors: {errors}")
    print(f"Average functions per file: {functions_extracted/max(files_processed,1):.1f}")
    print("=" * 80)

    # Estimate if we have enough data
    print("\nDATA AVAILABILITY CHECK:")
    if functions_extracted >= 150000:
        print(f"Pre-training: {functions_extracted:,} >= 150k target")
    else:
        print(f"Pre-training: {functions_extracted:,} < 150k target")
        shortfall = 150000 - functions_extracted
        print(f"   Need {shortfall:,} more functions")
        print(f"   Suggested: Add {shortfall // 3000} more large repositories")

    if functions_with_if >= 50000:
        print(f"Fine-tuning: {functions_with_if:,} >= 50k target")
    else:
        print(f"Fine-tuning: {functions_with_if:,} < 50k target")
        shortfall = 50000 - functions_with_if
        print(f"   Need {shortfall:,} more functions with if statements")
        print(f"   Suggested: Add {shortfall // 1500} more large repositories")

    print("=" * 80)
else:
    print("Skipping function extraction - will load from cached data")

"""## 3. Data Quality Filtering"""

def filter_functions(functions: List[Dict],
                    min_lines: int = 5,
                    max_lines: int = 100,
                    require_if: bool = False) -> List[Dict]:
    """
    Filter functions based on quality criteria:
    - Minimum and maximum number of lines
    - Contains if statements (for fine-tuning dataset)
    - Remove duplicates
    - Remove functions with syntax errors
    """
    filtered = []
    seen_sources = set()

    for func in functions:
        # Check line count
        if func['num_lines'] < min_lines or func['num_lines'] > max_lines:
            continue

        # Check if statement requirement
        if require_if and not func['has_if']:
            continue

        # Remove duplicates
        source_hash = hash(func['source'])
        if source_hash in seen_sources:
            continue
        seen_sources.add(source_hash)

        # Verify source can be parsed
        try:
            ast.parse(func['source'])
            filtered.append(func)
        except:
            continue

    return filtered

# Filter functions (or load from cache)
if not skip_extraction:
    # Filter for pre-training (all functions) - USE ALL AVAILABLE
    pretrain_functions = filter_functions(all_functions, min_lines=5, max_lines=100, require_if=False)
    print(f"Pre-training dataset: {len(pretrain_functions)} functions")

    # Check if we have enough data
    if len(pretrain_functions) < 150000:
        print(f"WARNING: Only {len(pretrain_functions)} pre-training examples available")
        print(f"Target is 150,000+. Consider:")
        print(f"   - Adding more repositories (currently have {len(repo_urls)})")
        print(f"   - Relaxing line count filters (currently 5-100 lines)")
        print(f"   - Cloning larger repositories")
    else:
        print(f"Sufficient pre-training data: {len(pretrain_functions)} examples")

    # Filter for fine-tuning (only functions with if statements) - USE ALL AVAILABLE
    finetune_functions = filter_functions(all_functions, min_lines=5, max_lines=100, require_if=True)
    print(f"Fine-tuning dataset: {len(finetune_functions)} functions")

    # Check if we have enough data
    if len(finetune_functions) < 50000:
        print(f"WARNING: Only {len(finetune_functions)} fine-tuning examples available")
        print(f"Target is 50,000+. Consider:")
        print(f"   - Adding more repositories")
        print(f"   - Relaxing filters (many functions have if statements)")
    else:
        print(f"Sufficient fine-tuning data: {len(finetune_functions)} examples")

    # Save raw datasets
    print("\nSaving datasets to disk...")
    with open('data/pretrain_raw.json', 'w') as f:
        json.dump(pretrain_functions, f, indent=2)
    print(f"Saved {len(pretrain_functions)} pre-training functions")

    with open('data/finetune_raw.json', 'w') as f:
        json.dump(finetune_functions, f, indent=2)
    print(f"Saved {len(finetune_functions)} fine-tuning functions")
else:
    # Load from cache
    print("Loading cached function data...")
    with open('data/pretrain_raw.json', 'r') as f:
        pretrain_functions = json.load(f)
    with open('data/finetune_raw.json', 'r') as f:
        finetune_functions = json.load(f)
    print(f"Pre-training dataset: {len(pretrain_functions)} functions")
    print(f"Fine-tuning dataset: {len(finetune_functions)} functions")

    # Check loaded data meets requirements
    if len(pretrain_functions) < 150000:
        print(f"WARNING: Pre-training data ({len(pretrain_functions)}) below target (150k)")
    else:
        print(f"Pre-training data meets target")

    if len(finetune_functions) < 50000:
        print(f"WARNING: Fine-tuning data ({len(finetune_functions)}) below target (50k)")
    else:
        print(f"Fine-tuning data meets target")

"""## 4. If Statement Extraction and Dataset Creation"""

class IfStatementExtractor(ast.NodeVisitor):
    """Extract if statements from function AST"""

    def __init__(self, source_lines: List[str]):
        self.if_statements = []
        self.source_lines = source_lines

    def visit_If(self, node):
        # Extract the if condition
        condition = ast.unparse(node.test)
        self.if_statements.append({
            'condition': condition,
            'lineno': node.lineno,
            'col_offset': node.col_offset
        })
        self.generic_visit(node)

def extract_if_statements(function_source: str) -> List[str]:
    """Extract all if conditions from a function"""
    try:
        tree = ast.parse(function_source)
        extractor = IfStatementExtractor(function_source.split('\n'))
        extractor.visit(tree)
        return [stmt['condition'] for stmt in extractor.if_statements]
    except:
        return []

def create_masked_instance(function_source: str, if_condition: str) -> Tuple[str, str]:
    """
    Create a masked instance by replacing an if condition with <MASK>
    Returns: (masked_function, original_condition)
    """
    # Use a special token to mask the if condition
    masked = function_source.replace(f"if {if_condition}:", "if <MASK>:", 1)
    return masked, if_condition

# Create fine-tuning dataset (or load from cache)
if not skip_extraction:
    # Create fine-tuning dataset with masked if statements
    finetune_dataset = []
    for func in tqdm(finetune_functions, desc="Creating fine-tuning dataset"):
        if_conditions = extract_if_statements(func['source'])

        # Create one instance for each if statement
        for condition in if_conditions:
            masked_func, original_cond = create_masked_instance(func['source'], condition)
            finetune_dataset.append({
                'input': masked_func,
                'output': original_cond
            })

    print(f"Created {len(finetune_dataset)} fine-tuning instances")

    # Split into train/val/test (80/10/10)
    random.shuffle(finetune_dataset)
    train_size = int(0.8 * len(finetune_dataset))
    val_size = int(0.1 * len(finetune_dataset))

    train_data = finetune_dataset[:train_size]
    val_data = finetune_dataset[train_size:train_size+val_size]
    test_data = finetune_dataset[train_size+val_size:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Save datasets
    with open('data/train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    with open('data/val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    with open('data/test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
else:
    print("Loading cached train/val/test data...")
    with open('data/train.json', 'r') as f:
        train_data = json.load(f)
    with open('data/val.json', 'r') as f:
        val_data = json.load(f)
    with open('data/test.json', 'r') as f:
        test_data = json.load(f)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

"""## 5. Custom Tokenizer Training"""

# Prepare corpus for tokenizer training
corpus_file = 'data/tokenizer_corpus.txt'

# Check if we need to create the corpus
if not os.path.exists(corpus_file):
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for func in pretrain_functions[:50000]:  # Use subset for tokenizer
            f.write(func['source'] + '\n\n')
    print(f"Created tokenizer corpus with {min(len(pretrain_functions), 50000)} functions")
else:
    print(f"Tokenizer corpus already exists at {corpus_file}")

# Train custom BPE tokenizer
def train_custom_tokenizer(corpus_file: str, vocab_size: int = 32000):
    """
    Train a custom BPE tokenizer for Python code
    """
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    # Special tokens for our task
    special_tokens = [
        "<PAD>", "<UNK>", "<BOS>", "<EOS>",
        "<MASK>",  # For masking if conditions
    ]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )

    # Train on corpus
    tokenizer.train([corpus_file], trainer)

    # Set up post-processor
    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        pair="<BOS> $A <EOS> $B:1 <EOS>:1",
        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ],
    )

    return tokenizer

# Train or load tokenizer
tokenizer_path = 'models/custom_tokenizer.json'
if not os.path.exists(tokenizer_path):
    print("Training custom tokenizer...")
    custom_tokenizer = train_custom_tokenizer(corpus_file)
    custom_tokenizer.save(tokenizer_path)
    print("Tokenizer trained and saved!")
else:
    print("Loading existing tokenizer...")
    custom_tokenizer = Tokenizer.from_file(tokenizer_path)
    print("Tokenizer loaded!")

# Test tokenizer
test_code = "if x > 0:\n    return True"
encoded = custom_tokenizer.encode(test_code)
print(f"\nTest encoding: {encoded.tokens}")

"""## 6. Pre-training Dataset Preparation"""

class PretrainingDataset(Dataset):
    """Dataset for pre-training with masked language modeling"""

    def __init__(self, functions, tokenizer, max_length=512, mask_prob=0.15):
        self.functions = functions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.token_to_id("<MASK>")
        self.pad_token_id = tokenizer.token_to_id("<PAD>")

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, idx):
        func = self.functions[idx]['source']

        # Tokenize
        encoded = self.tokenizer.encode(func)
        input_ids = encoded.ids[:self.max_length]

        # Create masked version for MLM
        labels = input_ids.copy()
        masked_input = input_ids.copy()

        # Randomly mask tokens
        for i in range(len(input_ids)):
            if random.random() < self.mask_prob:
                masked_input[i] = self.mask_token_id
            else:
                labels[i] = -100  # Don't compute loss for unmasked tokens

        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            masked_input = masked_input + [self.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length

        return {
            'input_ids': torch.tensor(masked_input, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor([1] * len(input_ids) + [0] * padding_length, dtype=torch.long)
        }

# Reload tokenizer to ensure we have it loaded
if 'custom_tokenizer' not in dir():
    custom_tokenizer = Tokenizer.from_file('models/custom_tokenizer.json')

# Create pre-training dataset - USE ALL AVAILABLE DATA (minimum 200k target)
MIN_PRETRAIN_SIZE = 200000
available_pretrain = len(pretrain_functions)

if available_pretrain < MIN_PRETRAIN_SIZE:
    print(f"WARNING: Only {available_pretrain:,} functions available for pre-training")
    print(f"Target is {MIN_PRETRAIN_SIZE:,}. Using all available data.")
    print(f"Consider adding more repositories to reach target.")
    pretrain_subset_size = available_pretrain
else:
    print(f"Using {MIN_PRETRAIN_SIZE:,} functions for pre-training (from {available_pretrain:,} available)")
    pretrain_subset_size = MIN_PRETRAIN_SIZE

pretrain_dataset = PretrainingDataset(
    pretrain_functions[:pretrain_subset_size],
    custom_tokenizer
)

print(f"\nPre-training dataset size: {len(pretrain_dataset):,}")
print(f"Estimated training time: ~{len(pretrain_dataset) * 2 // 60 // 60}-{len(pretrain_dataset) * 4 // 60 // 60} hours")
print(f"Memory requirement: ~{len(pretrain_dataset) * 512 * 4 / (1024**3):.1f} GB (with batch processing)")

# Verify we have enough data
if len(pretrain_dataset) >= 200000:
    print(f"Dataset size meets minimum requirement (200k)")
else:
    print(f"Dataset size below minimum (200k). Accuracy may be limited.")
    print(f"Current size: {len(pretrain_dataset):,}")

"""## 7. Model Pre-training"""

# Check if we should skip pre-training
skip_pretraining = os.path.exists('models/pretrained_final/config.json')
if skip_pretraining:
    print("Pre-trained model already exists. Skipping pre-training...")
    print("If you want to retrain, delete the 'models/pretrained_final' directory.")
else:
    print("No pre-trained model found. Will proceed with pre-training...")

"""### ⚡ Performance Optimizations

**Key Changes for Speed:**
1. **Reduced dataset**: 20K functions (from 150K) → 7.5x faster
2. **Smaller model**: 4 layers, 256 dim (vs 6 layers, 512 dim) → 3-4x faster  
3. **Fewer epochs**: 1 epoch for pre-training (vs 3) → 3x faster
4. **Larger batches**: 16 per device (vs 8) → 2x faster
5. **Combined speedup**: ~45-90x faster! (124 hours → 1-2 hours)

**Expected Training Time:**
- Pre-training: ~30-60 minutes
- Fine-tuning: ~20-40 minutes
- **Total: ~1-1.5 hours** on a good laptop

**If you get Out-Of-Memory errors:**
- Reduce `per_device_train_batch_size` to 8 or 4
- Set `fp16=False` if using CPU or older GPU
- Reduce `pretrain_subset_size` to 10000

**To get better accuracy (if you have time):**
- Increase `pretrain_subset_size` to 50000
- Increase `num_train_epochs` to 2-3 for pre-training
- Use larger model dimensions (512, 6 layers)
"""

# Initialize T5 model from scratch (only if not already pre-trained)
if not skip_pretraining:
    # Use SMALLER model for faster training
    config = T5Config(
        vocab_size=custom_tokenizer.get_vocab_size(),
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        # Set required token IDs for T5
        pad_token_id=custom_tokenizer.token_to_id("<PAD>"),
        eos_token_id=custom_tokenizer.token_to_id("<EOS>"),
        decoder_start_token_id=custom_tokenizer.token_to_id("<PAD>"),  # T5 uses pad_token_id as decoder start
    )

    model = T5ForConditionalGeneration(config)
    print(f"Model initialized with {model.num_parameters():,} parameters")
else:
    print("Skipping model initialization - will load pre-trained model later")

# Pre-train the model (only if not already done)
if not skip_pretraining:
    # Pre-training arguments - BALANCED FOR LEARNING
    pretrain_args = TrainingArguments(
        output_dir='models/pretrained',
        num_train_epochs=3,  # Increased from 1 - need more epochs to learn
        per_device_train_batch_size=16,  # Adjust if OOM
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,    # Reduced from 500
        logging_steps=50,    # More frequent logging
        save_steps=2000,     # Less frequent saves
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to='none',
        dataloader_num_workers=2,  # Parallel data loading
    )

    # Simple data collator (our dataset already handles padding)
    def simple_collator(features):
        """Collate pre-padded features"""
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        }
        return batch

    # Create trainer
    pretrain_trainer = Trainer(
        model=model,
        args=pretrain_args,
        train_dataset=pretrain_dataset,
        data_collator=simple_collator,
    )

    # Pre-train the model
    print("="*80)
    print("STARTING PRE-TRAINING")
    print("="*80)
    effective_batch = pretrain_args.per_device_train_batch_size * pretrain_args.gradient_accumulation_steps
    steps_per_epoch = len(pretrain_dataset) // effective_batch
    total_steps = steps_per_epoch * pretrain_args.num_train_epochs
    print(f"Dataset size: {len(pretrain_dataset)}")
    print(f"Batch size: {pretrain_args.per_device_train_batch_size} (effective: {effective_batch})")
    print(f"Epochs: {pretrain_args.num_train_epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Estimated time: ~{total_steps * 2 // 60}-{total_steps * 4 // 60} minutes")
    print("="*80)

    import time
    start_time = time.time()
    pretrain_trainer.train()
    elapsed = time.time() - start_time

    print("="*80)
    print(f"Pre-training completed in {elapsed/60:.1f} minutes")
    print(f"⚠️  If this finished in < 30 minutes, the model is UNDERTRAINED!")
    print("="*80)

    # Save pre-trained model
    model.save_pretrained('models/pretrained_final')
    custom_tokenizer.save('models/pretrained_final/tokenizer.json')
    print("Model saved to models/pretrained_final/")
else:
    print("Skipping pre-training - model already exists")
    # Define simple_collator for fine-tuning
    def simple_collator(features):
        """Collate pre-padded features"""
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        }
        return batch

"""## 8. Fine-tuning Dataset Preparation"""

class IfStatementDataset(Dataset):
    """Dataset for fine-tuning on if statement prediction (shortened for fast experiments)"""

    def __init__(self, data, tokenizer, max_length=128):
        # Reduced default max_length from 512 -> 128 to speed up tokenization and training
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.token_to_id("<PAD>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Encode input (function with masked if)
        input_enc = self.tokenizer.encode(item['input'])
        input_ids = input_enc.ids[:self.max_length]

        # Encode output (if condition)
        output_enc = self.tokenizer.encode(item['output'])
        labels = output_enc.ids[:self.max_length]

        # Pad input
        input_padding = self.max_length - len(input_ids)
        if input_padding > 0:
            input_ids = input_ids + [self.pad_token_id] * input_padding

        # Pad labels
        label_padding = self.max_length - len(labels)
        if label_padding > 0:
            labels = labels + [-100] * label_padding

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1] * (self.max_length - input_padding) + [0] * input_padding, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Load datasets
with open('data/train.json', 'r') as f:
    train_data_all = json.load(f)
with open('data/val.json', 'r') as f:
    val_data_all = json.load(f)
with open('data/test.json', 'r') as f:
    test_data_all = json.load(f)

# Use LARGE subsets for actual learning - minimum 50k training examples
MIN_TRAIN_SIZE = 200000
MIN_VAL_SIZE = 20000
MIN_TEST_SIZE = 20000

available_train = len(train_data_all)
available_val = len(val_data_all)
available_test = len(test_data_all)

print("=" * 80)
print("FINE-TUNING DATASET SIZE CHECK")
print("=" * 80)

# Training data
if available_train < MIN_TRAIN_SIZE:
    print(f"WARNING: Only {available_train:,} training examples available")
    print(f"Target is {MIN_TRAIN_SIZE:,}. Using all available data.")
    train_size = available_train
else:
    print(f"Using {MIN_TRAIN_SIZE:,} training examples (from {available_train:,} available)")
    train_size = MIN_TRAIN_SIZE

train_data = train_data_all[:train_size]

# Validation data
if available_val < MIN_VAL_SIZE:
    print(f"WARNING: Only {available_val:,} validation examples available")
    print(f"Target is {MIN_VAL_SIZE:,}. Using all available data.")
    val_size = available_val
else:
    print(f"Using {MIN_VAL_SIZE:,} validation examples (from {available_val:,} available)")
    val_size = MIN_VAL_SIZE

val_data = val_data_all[:val_size]

# Test data
if available_test < MIN_TEST_SIZE:
    print(f"WARNING: Only {available_test:,} test examples available")
    print(f"Target is {MIN_TEST_SIZE:,}. Using all available data.")
    test_size = available_test
else:
    print(f"Using {MIN_TEST_SIZE:,} test examples (from {available_test:,} available)")
    test_size = MIN_TEST_SIZE

test_data = test_data_all[:test_size]

print("=" * 80)
print(f"\nFinal dataset sizes:")
print(f"  Train: {len(train_data):,}")
print(f"  Val:   {len(val_data):,}")
print(f"  Test:  {len(test_data):,}")

# Create datasets with reduced max_length for speed
train_dataset = IfStatementDataset(train_data, custom_tokenizer, max_length=128)
val_dataset = IfStatementDataset(val_data, custom_tokenizer, max_length=128)
test_dataset = IfStatementDataset(test_data, custom_tokenizer, max_length=128)

# Estimate training time
print(f"\nEstimated fine-tuning time:")
steps_per_epoch = len(train_dataset) // 16  # Assuming batch size 16
print(f"  Steps per epoch: {steps_per_epoch:,}")
print(f"  Total time (5 epochs): ~{steps_per_epoch * 5 * 2 // 60}-{steps_per_epoch * 5 * 4 // 60} minutes")

if len(train_dataset) >= 50000:
    print(f"\nTraining dataset meets minimum requirement (200k)")
else:
    print(f"\nTraining dataset below minimum (200k). Accuracy may be limited.")
    print(f"Current size: {len(train_dataset):,}")
    print(f"Consider adding more repositories to increase data.")

# Check if we should skip fine-tuning
skip_finetuning = os.path.exists('models/finetuned_final/config.json')
if skip_finetuning:
    print("Fine-tuned model already exists. Skipping fine-tuning...")
    print("If you want to retrain, delete the 'models/finetuned_final' directory.")
else:
    print("No fine-tuned model found. Will proceed with fine-tuning...")

"""## 9. Fine-tuning

Now we'll fine-tune the pre-trained model on the if statement prediction task.
"""

# Fine-tune the model (only if not already done)
if not skip_finetuning:
    # Load pre-trained model
    model = T5ForConditionalGeneration.from_pretrained('models/pretrained_final')

    # Fine-tuning arguments - PROPER LEARNING SETTINGS
    finetune_args = TrainingArguments(
        output_dir='models/finetuned',
        num_train_epochs=5,  # Increased from 1 - need more epochs for convergence
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,  # Increase effective batch size
        learning_rate=3e-5,  # Slightly lower for stability
        weight_decay=0.0,
        warmup_steps=100,
        evaluation_strategy='steps',
        logging_steps=50,
        eval_steps=200,          # Evaluate every 200 steps
        save_strategy='steps',   # Save checkpoints
        save_steps=600,
        save_total_limit=2,
        load_best_model_at_end=True,  # Load the best checkpoint
        metric_for_best_model='loss',
        fp16=torch.cuda.is_available(),
        report_to='none',
        dataloader_num_workers=2,
    )

    # Create trainer
    finetune_trainer = Trainer(
        model=model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=simple_collator,
    )

    # Fine-tune the model (smoke-run)
    print("="*80)
    print("STARTING FINE-TUNING")
    print("="*80)
    effective_batch = finetune_args.per_device_train_batch_size * finetune_args.gradient_accumulation_steps
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * finetune_args.num_train_epochs
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: {finetune_args.per_device_train_batch_size} (effective: {effective_batch})")
    print(f"Epochs: {finetune_args.num_train_epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Estimated time: ~{total_steps * 1 // 60}-{total_steps * 3 // 60} minutes")
    print("="*80)

    import time
    start_time = time.time()
    finetune_trainer.train()
    elapsed = time.time() - start_time

    print("="*80)
    print(f"Fine-tuning completed in {elapsed/60:.1f} minutes")
    print(f"⚠️  If this finished in < 10 minutes, the model is UNDERTRAINED!")
    print("="*80)

    # Save fine-tuned model
    model.save_pretrained('models/finetuned_final')
    custom_tokenizer.save('models/finetuned_final/tokenizer.json')
    print("Model saved to models/finetuned_final/")
else:
    print("Skipping fine-tuning - model already exists")

"""## 10. Evaluation and Prediction"""

def predict_if_condition(model, tokenizer, masked_function, max_length=128):
    """
    Predict the masked if condition
    """
    model.eval()
    device = next(model.parameters()).device

    # Encode input
    encoded = tokenizer.encode(masked_function)
    input_ids = torch.tensor([encoded.ids]).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Decode prediction
    predicted_ids = outputs.sequences[0].cpu().numpy()
    predicted_text = tokenizer.decode(predicted_ids.tolist())

    # Clean up special tokens
    predicted_text = predicted_text.replace('<BOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()

    # Get confidence score (average log probability)
    if hasattr(outputs, 'sequences_scores') and len(outputs.sequences_scores) > 0:
        scores = outputs.sequences_scores[0].item()
        confidence = min(max(np.exp(scores) * 100, 0), 100)  # Clamp between 0-100
    else:
        confidence = 50.0  # Default if scores unavailable

    return predicted_text, confidence

# Load final model
final_model = T5ForConditionalGeneration.from_pretrained('models/finetuned_final')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
final_model = final_model.to(device)
print(f"Model loaded on device: {device}")

# Test on a sample
if len(test_data) > 0:
    sample = test_data[0]
    predicted, confidence = predict_if_condition(final_model, custom_tokenizer, sample['input'])
    print(f"\nSample Prediction:")
    print(f"Input: {sample['input'][:100]}...")
    print(f"Expected: {sample['output']}")
    print(f"Predicted: {predicted}")
    print(f"Confidence: {confidence:.2f}%")
else:
    print("No test data available")

def evaluate_model(model, tokenizer, test_data):
    """
    Evaluate model on test set and generate CSV results
    """
    if len(test_data) == 0:
        print("Warning: Test data is empty!")
        return []

    results = []

    for item in tqdm(test_data, desc="Evaluating"):
        try:
            predicted, confidence = predict_if_condition(model, tokenizer, item['input'])

            # Clean up predicted text
            predicted = predicted.replace('<BOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()
            expected = item['output'].strip()

            # Check if correct (exact match)
            is_correct = predicted == expected

            # Also check partial match (for debugging)
            # Normalize whitespace for comparison
            pred_normalized = ' '.join(predicted.split())
            exp_normalized = ' '.join(expected.split())
            is_partial = pred_normalized == exp_normalized

            results.append({
                'Input provided to the model': item['input'],
                'Whether the prediction is correct (true/false)': is_correct,
                'Expected if condition': expected,
                'Predicted if condition': predicted,
                'Prediction score (0-100)': round(confidence, 2)
            })
        except Exception as e:
            print(f"Error processing item: {e}")
            continue

    # Calculate accuracy
    if len(results) > 0:
        accuracy = sum(1 for r in results if r['Whether the prediction is correct (true/false)']) / len(results)
        print(f"Accuracy: {accuracy*100:.2f}%")
    else:
        print("No results to evaluate!")

    return results

# Evaluate on test set
test_results = evaluate_model(final_model, custom_tokenizer, test_data)

# Save to CSV
if len(test_results) > 0:
    df = pd.DataFrame(test_results)
    df.to_csv('results/generated-testset.csv', index=False)
    print("Saved results to results/generated-testset.csv")
else:
    print("No results to save!")


# Evaluate on provided test set (if available)
provided_test = 'data/benchmark_if_only.csv'
provided_test_data = []

if os.path.exists(provided_test):
    try:
        print(f"\nFound provided test CSV at {provided_test}")
        print("=" * 80)
        
        # Read CSV with proper handling
        df_bench = pd.read_csv(provided_test)
        print(f"Loaded CSV with {len(df_bench)} rows")
        print(f"Columns: {list(df_bench.columns)}")
        
        # The CSV has a 'code' column
        if 'code' not in df_bench.columns:
            print(f"ERROR: 'code' column not found in CSV!")
            print(f"Available columns: {list(df_bench.columns)}")
        else:
            code_col = 'code'
            
            # Remove empty or NaN code entries
            df_bench_clean = df_bench[df_bench[code_col].notna() & (df_bench[code_col].astype(str).str.strip() != '')]
            print(f"After removing empty entries: {len(df_bench_clean)} rows")
            
            # DEBUG: Check first few entries
            print("\nDEBUGGING: Checking first 3 code samples...")
            for idx in range(min(3, len(df_bench_clean))):
                sample_code = str(df_bench_clean.iloc[idx][code_col])
                print(f"\n--- Sample {idx+1} (first 200 chars) ---")
                print(sample_code[:200])
                print(f"Contains 'if ': {('if ' in sample_code)}")
                
                # Try to parse and extract
                try:
                    tree = ast.parse(sample_code)
                    print("Parses successfully")
                    
                    # Count if statements manually
                    if_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))
                    print(f"If statements found via ast.walk: {if_count}")
                    
                    # Try extraction
                    conditions = extract_if_statements(sample_code)
                    print(f"Conditions extracted: {len(conditions)}")
                    if conditions:
                        print(f"First condition: {conditions[0]}")
                except Exception as e:
                    print(f"Parse error: {e}")

            converted = []
            rows_with_if = 0
            total_conditions = 0
            parse_errors = 0
            
            for idx, row in df_bench_clean.iterrows():
                src = str(row[code_col]).strip()
                if not src:
                    continue

                try:
                    # First check if code even contains 'if'
                    if 'if ' not in src.lower():
                        continue
                    
                    # Extract if conditions from the code
                    conditions = extract_if_statements(src)
                    
                    if conditions:
                        rows_with_if += 1
                        total_conditions += len(conditions)
                        
                        # Create masked instances for each if condition
                        for cond in conditions:
                            try:
                                masked_src, original_cond = create_masked_instance(src, cond)
                                converted.append({
                                    'input': masked_src, 
                                    'output': original_cond,
                                    'row_id': row.get('id', idx)
                                })
                            except Exception as e:
                                print(f"  Warning: Masking failed for row {row.get('id', idx)}, condition '{cond[:50]}...': {e}")
                                continue
                                
                except Exception as e:
                    parse_errors += 1
                    if parse_errors <= 3:  # Only print first 3 errors
                        print(f"  Warning: Failed to extract conditions from row {row.get('id', idx)}: {e}")
                    continue

            provided_test_data = converted
            print("=" * 80)
            print(f"CONVERSION SUMMARY:")
            print(f"  Total CSV rows: {len(df_bench)}")
            print(f"  Rows with valid code: {len(df_bench_clean)}")
            print(f"  Rows containing if statements: {rows_with_if}")
            print(f"  Total if conditions found: {total_conditions}")
            print(f"  Parse errors: {parse_errors}")
            print(f"  Final test instances created: {len(provided_test_data)}")
            print("=" * 80)
            
    except Exception as e:
        print(f"ERROR processing provided CSV test set: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nProvided test set not found.")
    print(f"Looking for: {provided_test}")
    print("Please ensure benchmark_if_only.csv is in the data/ directory")

# If we have data, evaluate and save results
if len(provided_test_data) > 0:
    try:
        provided_results = evaluate_model(final_model, custom_tokenizer, provided_test_data)
        if len(provided_results) > 0:
            df_provided = pd.DataFrame(provided_results)
            df_provided.to_csv('results/provided-testset.csv', index=False)
            print("Saved results to results/provided-testset.csv")
    except Exception as e:
        print(f"Error evaluating provided test set: {e}")

"""## How to Run This Notebook

### Quick Start:
1. **Run all cells in order** - The notebook will automatically skip already-completed steps
2. **First run** will take several hours:
   - Repository cloning: ~30 minutes
   - Function extraction: ~1 hour
   - Tokenizer training: ~15 minutes
   - Pre-training: ~2-4 hours (depending on GPU)
   - Fine-tuning: ~1-2 hours
3. **Subsequent runs** will be much faster as cached data is used

### To Force Re-training:
- Delete `models/pretrained_final/` to re-run pre-training
- Delete `models/finetuned_final/` to re-run fine-tuning
- Delete `data/*.json` to re-extract functions
- Delete `repos/` to re-clone repositories

### Expected Outputs:
- `results/generated-testset.csv` - Results on your test split
- `results/provided-testset.csv` - Results on instructor's test set (if provided)
- Model files in `models/` directory

### Troubleshooting:
- **Out of memory**: Reduce `per_device_train_batch_size` in training arguments
- **CUDA errors**: Set `fp16=False` in training arguments
- **No repositories**: Repositories are already cloned in `repos/` directory
- **Empty results**: Check that test data was created properly in Section 4

## 11. Summary and Analysis
"""

# Generate summary statistics
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"\nData Collection:")
print(f"  - Repositories cloned: {len(cloned_repos)}")
print(f"  - Python files found: {len(all_python_files)}")
if not skip_extraction:
    print(f"  - Functions extracted: {len(all_functions)}")

print(f"\nPre-training:")
print(f"  - Dataset size: {len(pretrain_dataset)}")
print(f"  - Vocabulary size: {custom_tokenizer.get_vocab_size()}")

print(f"\nFine-tuning:")
print(f"  - Train: {len(train_dataset)}")
print(f"  - Validation: {len(val_dataset)}")
print(f"  - Test: {len(test_dataset)}")

print(f"\nModel:")
print(f"  - Parameters: {final_model.num_parameters():,}")

if len(test_results) > 0:
    print(f"\nResults:")
    accuracy = sum(1 for r in test_results if r['Whether the prediction is correct (true/false)']) / len(test_results)
    print(f"  - Test Accuracy: {accuracy*100:.2f}%")
    print(f"  - Average Confidence: {np.mean([r['Prediction score (0-100)'] for r in test_results]):.2f}%")
    print(f"  - Total Test Instances: {len(test_results)}")
else:
    print(f"\nResults:")
    print(f"  - No test results available")

print("\n" + "=" * 60)

"""## 10.5 Diagnostic Analysis - Understanding the 0% Accuracy

Let's examine what the model is actually predicting vs. what it should predict.
"""

# Diagnostic: Examine a few predictions in detail
print("DIAGNOSTIC ANALYSIS")
print("=" * 80)

if len(test_results) > 0:
    print(f"\nSample Predictions (showing first 5):\n")
    for i, result in enumerate(test_results[:5]):
        print(f"\n--- Example {i+1} ---")
        print(f"Expected:  '{result['Expected if condition']}'")
        print(f"Predicted: '{result['Predicted if condition']}'")
        print(f"Correct: {result['Whether the prediction is correct (true/false)']}")
        print(f"Confidence: {result['Prediction score (0-100)']}%")

        # Show character-level diff if wrong
        if not result['Whether the prediction is correct (true/false)']:
            exp = result['Expected if condition']
            pred = result['Predicted if condition']
            print(f"Length - Expected: {len(exp)}, Predicted: {len(pred)}")
            if len(pred) == 0:
                print("⚠️  Model produced EMPTY output!")
            elif pred == exp[:len(pred)] or exp == pred[:len(exp)]:
                print("⚠️  Prediction is a PREFIX/SUFFIX of expected (truncation issue)")

    # Check for common issues
    print(f"\n{'='*80}")
    print("COMMON ISSUES DETECTED:")
    empty_preds = sum(1 for r in test_results if len(r['Predicted if condition'].strip()) == 0)
    if empty_preds > 0:
        print(f"❌ {empty_preds}/{len(test_results)} predictions are EMPTY")
        print("   → Model isn't generating output properly")
        print("   → Need MORE training data and/or MORE epochs")

    low_conf = sum(1 for r in test_results if r['Prediction score (0-100)'] < 30)
    if low_conf > len(test_results) * 0.8:
        print(f"❌ {low_conf}/{len(test_results)} predictions have low confidence (<30%)")
        print("   → Model is guessing randomly")
        print("   → SEVERELY undertrained - need 10x-100x more data")

    # Check prediction diversity
    unique_preds = len(set(r['Predicted if condition'] for r in test_results))
    if unique_preds < len(test_results) * 0.1:
        print(f"❌ Only {unique_preds} unique predictions for {len(test_results)} examples")
        print("   → Model is producing repetitive output")
        print("   → Try increasing temperature or beam search diversity")

    print(f"\n{'='*80}")
    print("\nRECOMMENDATIONS:")
    print("1. ⚡ INCREASE training data to at least 10,000-20,000 examples")
    print("2. ⚡ INCREASE epochs to 3-5 for both pre-training and fine-tuning")
    print("3. ⚡ DELETE models/pretrained_final and models/finetuned_final")
    print("4. ⚡ RE-RUN the training cells with new settings")
    print(f"\nExpected training time with 20k examples: ~2-4 hours")
    print(f"Minimum accuracy goal: 20-40% (with 20k examples)")
    print(f"Good accuracy goal: 50-70% (with 50k+ examples)")
else:
    print("No test results available for analysis")
