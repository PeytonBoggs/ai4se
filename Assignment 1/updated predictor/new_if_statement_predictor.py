"""
Streamlined If Statement Predictor using CodeT5-small
Trains a model to predict masked if conditions in Python code
"""

import os
import ast
import json
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import git

import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# 1. REPOSITORY CLONING
# ============================================================================

REPOS = [
    "https://github.com/psf/requests",
    "https://github.com/pallets/flask",
    "https://github.com/django/django",
    "https://github.com/tiangolo/fastapi",
    "https://github.com/numpy/numpy",
    "https://github.com/pandas-dev/pandas",
    "https://github.com/scikit-learn/scikit-learn",
    "https://github.com/pytorch/pytorch",
    "https://github.com/tensorflow/tensorflow",
    "https://github.com/huggingface/transformers",
    "https://github.com/ansible/ansible",
    "https://github.com/scrapy/scrapy",
    "https://github.com/sqlalchemy/sqlalchemy",
    "https://github.com/python-pillow/Pillow",
]

def clone_repos():
    cloned = []
    for url in tqdm(REPOS, desc="Cloning repos"):
        name = url.split('/')[-1]
        path = f'repos/{name}'
        if not os.path.exists(path):
            try:
                git.Repo.clone_from(url, path, depth=1)
            except:
                pass
        if os.path.exists(path):
            cloned.append(path)
    return cloned

# ============================================================================
# 2. FUNCTION EXTRACTION
# ============================================================================

class FunctionExtractor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
    
    def visit_FunctionDef(self, node):
        source = ast.unparse(node)
        has_if = any(isinstance(n, ast.If) for n in ast.walk(node))
        self.functions.append({
            'source': source,
            'num_lines': len(source.split('\n')),
            'has_if': has_if
        })
        self.generic_visit(node)

def extract_functions(repo_paths):
    all_funcs = []
    for repo in tqdm(repo_paths, desc="Extracting functions"):
        for root, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if d not in ['test', 'tests', '__pycache__', '.git']]
            for file in files:
                if file.endswith('.py'):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            tree = ast.parse(f.read())
                            ext = FunctionExtractor()
                            ext.visit(tree)
                            all_funcs.extend(ext.functions)
                    except:
                        pass
    return all_funcs

def filter_functions(funcs, min_lines=5, max_lines=100, require_if=False):
    filtered = []
    seen = set()
    for f in funcs:
        if f['num_lines'] < min_lines or f['num_lines'] > max_lines:
            continue
        if require_if and not f['has_if']:
            continue
        h = hash(f['source'])
        if h in seen:
            continue
        seen.add(h)
        try:
            ast.parse(f['source'])
            filtered.append(f)
        except:
            pass
    return filtered

# ============================================================================
# 3. IF STATEMENT MASKING
# ============================================================================

def extract_if_conditions(source):
    conditions = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                conditions.append(ast.unparse(node.test))
    except:
        pass
    return conditions

def create_dataset(functions):
    data = []
    for func in tqdm(functions, desc="Creating dataset"):
        conditions = extract_if_conditions(func['source'])
        for cond in conditions:
            masked = func['source'].replace(f"if {cond}:", "if <MASK>:", 1)
            data.append({'input': masked, 'output': cond})
    return data

# ============================================================================
# 4. DATASET CLASS
# ============================================================================

class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item['input'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = self.tokenizer(
            item['output'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels['input_ids'][labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

# ============================================================================
# 5. MAIN PIPELINE
# ============================================================================

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load or extract data
    if os.path.exists('data/train.json'):
        print("Loading cached data...")
        with open('data/train.json') as f: train_data = json.load(f)
        with open('data/val.json') as f: val_data = json.load(f)
        with open('data/test.json') as f: test_data = json.load(f)
    else:
        print("Extracting data from repositories...")
        repos = clone_repos()
        all_funcs = extract_functions(repos)
        
        # Split into pretrain and finetune
        pretrain_funcs = filter_functions(all_funcs, require_if=False)
        finetune_funcs = filter_functions(all_funcs, require_if=True)
        
        print(f"Pretrain: {len(pretrain_funcs)}, Finetune: {len(finetune_funcs)}")
        
        # Create masked dataset
        dataset = create_dataset(finetune_funcs)
        random.shuffle(dataset)
        
        # 80/10/10 split
        n = len(dataset)
        train_data = dataset[:int(0.8*n)]
        val_data = dataset[int(0.8*n):int(0.9*n)]
        test_data = dataset[int(0.9*n):]

        # Keep training data small to speed up training
        if len(train_data) > 20000:
            train_data = train_data[:20000]
            print(f"Using subset: {len(train_data)} examples")
        
        # Save
        with open('data/train.json', 'w') as f: json.dump(train_data, f)
        with open('data/val.json', 'w') as f: json.dump(val_data, f)
        with open('data/test.json', 'w') as f: json.dump(test_data, f)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Load CodeT5 model and tokenizer
    print("Loading CodeT5-small model...")
    model_name = "Salesforce/codet5-small"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Increase model size by scaling up dimensions
    config = model.config
    config.d_model = 1024  # Increased from 512
    config.d_ff = 4096     # Increased from 2048
    config.num_layers = 8  # Increased from 6
    config.num_decoder_layers = 8
    config.num_heads = 16  # Increased from 8
    
    # Reinitialize with larger config
    model = T5ForConditionalGeneration(config)
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Add special tokens
    special_tokens = {'additional_special_tokens': ['<MASK>']}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Create datasets
    train_dataset = CodeDataset(train_data, tokenizer)
    val_dataset = CodeDataset(val_data, tokenizer)
    test_dataset = CodeDataset(test_data, tokenizer)
    
    # Training argument
    training_args = TrainingArguments(
        output_dir='models/codet5_finetuned',
        num_train_epochs=3,  # Reduced from 5
        per_device_train_batch_size=32,  # Increased from 8
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,  # Reduced from 4
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=100,  # Reduced from 500
        eval_strategy='steps',
        eval_steps=2000,  # Increased from 500
        save_steps=5000,  # Increased from 1000
        save_total_limit=1,  # Reduced from 2
        load_best_model_at_end=False,  # Disabled for speed
        fp16=False,  # CPU doesn't support fp16
        logging_steps=500,  # Reduced logging
        report_to='none',
        dataloader_num_workers=8,  # Use multiple CPU cores
        dataloader_pin_memory=False,  # Disable for CPU
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    
    print("Training...")
    trainer.train()
    
    # Save model
    model.save_pretrained('models/codet5_final')
    tokenizer.save_pretrained('models/codet5_final')
    print("Model saved to models/codet5_final")
    
    # Evaluate
    print("Evaluating...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = []
    for item in tqdm(test_data, desc="Testing"):
        inputs = tokenizer(item['input'], return_tensors='pt', max_length=256, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256, num_beams=5)
        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            'Input provided to the model': item['input'],
            'Whether the prediction is correct (true/false)': predicted.strip() == item['output'].strip(),
            'Expected if condition': item['output'],
            'Predicted if condition': predicted,
            'Prediction score (0-100)': 100.0 if predicted.strip() == item['output'].strip() else 0.0
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/generated-testset.csv', index=False)
    
    accuracy = df['Whether the prediction is correct (true/false)'].mean()
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Results saved to results/generated-testset.csv")

if __name__ == '__main__':
    main()