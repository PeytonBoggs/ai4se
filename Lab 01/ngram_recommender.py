#!/usr/bin/env python3
"""
N-gram Code Recommender for Java Methods
AI4SE 2025 - Lab-01
Prof. Antonio Mastropaolo

This script implements an N-gram probabilistic language model for code completion.
"""

import pandas as pd
import numpy as np
import json
import re
import argparse
import logging
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import math
import random
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeTokenizer:
    """Tokenizes Java code into meaningful tokens."""
    
    def __init__(self):
        # Java keywords and common patterns
        self.java_keywords = {
            'public', 'private', 'protected', 'static', 'final', 'abstract',
            'class', 'interface', 'extends', 'implements', 'import', 'package',
            'if', 'else', 'while', 'for', 'do', 'switch', 'case', 'default',
            'try', 'catch', 'finally', 'throw', 'throws', 'return', 'break',
            'continue', 'new', 'this', 'super', 'null', 'true', 'false',
            'int', 'long', 'short', 'byte', 'char', 'boolean', 'float', 'double',
            'String', 'void', 'List', 'Map', 'Set', 'ArrayList', 'HashMap'
        }
        
        # Special tokens
        self.SPECIAL_TOKENS = {
            '<SOS>': '<SOS>',  # Start of sequence
            '<EOS>': '<EOS>',  # End of sequence
            '<UNK>': '<UNK>'   # Unknown token
        }
    
    def clean_code(self, code: str) -> str:
        """Clean and preprocess Java code."""
        if not code or pd.isna(code):
            return ""
        
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Remove excessive whitespace
        code = re.sub(r'\s+', ' ', code)
        
        # Remove empty lines
        code = '\n'.join(line.strip() for line in code.split('\n') if line.strip())
        
        return code.strip()
    
    def tokenize(self, code: str) -> List[str]:
        """Tokenize Java code into tokens."""
        if not code:
            return []
        
        # Clean the code first
        code = self.clean_code(code)
        if not code:
            return []
        
        # Enhanced tokenization pattern for better Java code handling
        # This pattern captures:
        # - String literals (with escaped quotes)
        # - Numbers (integers, floats, hex, binary)
        # - Java identifiers and keywords
        # - Multi-character operators
        # - Single character operators and punctuation
        # - Whitespace-separated tokens
        pattern = r'''
            "(?:[^"\\]|\\.)*"|                     # String literals
            '(?:[^'\\]|\\.)*'|                     # Character literals
            0[xX][0-9a-fA-F]+|                     # Hexadecimal numbers
            0[bB][01]+|                            # Binary numbers
            \d+\.?\d*[fFdD]?|                      # Numbers with optional suffix
            [a-zA-Z_$][a-zA-Z0-9_$]*|              # Java identifiers
            \+\+|--|==|!=|<=|>=|&&|\|\||           # Multi-char operators
            <<|>>|>>>|\+=|-=|\*=|/=|%=|           # Assignment operators
            [+\-*/=<>!&|^%~?:]+|                   # Other operators
            [{}();,.\[\]]|                         # Delimiters
            \S                                     # Any other non-whitespace
        '''
        
        tokens = re.findall(pattern, code, re.VERBOSE)
        
        # Filter out empty tokens and normalize
        tokens = [token.strip() for token in tokens if token.strip()]
        
        # Additional filtering for very common but less informative tokens
        filtered_tokens = []
        for token in tokens:
            # Keep meaningful tokens
            if len(token) > 0 and token not in {'\n', '\r', '\t'}:
                filtered_tokens.append(token)
        
        return filtered_tokens

class NGramModel:
    """N-gram probabilistic language model for code completion."""
    
    def __init__(self, n: int = 3, smoothing: str = 'laplace', min_count: int = 1):
        """
        Initialize N-gram model.
        
        Args:
            n: N-gram size (e.g., 3 for trigrams)
            smoothing: Smoothing technique ('laplace', 'good_turing', 'none')
            min_count: Minimum count threshold for n-grams
        """
        self.n = n
        self.smoothing = smoothing
        self.min_count = min_count
        self.tokenizer = CodeTokenizer()
        
        # N-gram counts
        self.ngram_counts = defaultdict(Counter)  # (n-1)-gram -> {next_token: count}
        self.context_counts = Counter()  # (n-1)-gram -> total_count
        self.vocabulary = set()
        self.total_tokens = 0
        
        logger.info(f"Initialized {n}-gram model with {smoothing} smoothing")
    
    def train(self, corpus: List[str]):
        """Train the N-gram model on a corpus of code."""
        logger.info(f"Training {self.n}-gram model on {len(corpus)} examples...")
        
        all_tokens = []
        
        # Tokenize all examples
        for i, code in enumerate(corpus):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(corpus)} examples")
            
            tokens = self.tokenizer.tokenize(code)
            if len(tokens) < self.n:  # Skip sequences too short
                continue
            
            # Add special tokens
            tokens = [self.tokenizer.SPECIAL_TOKENS['<SOS>']] + tokens + [self.tokenizer.SPECIAL_TOKENS['<EOS>']]
            all_tokens.extend(tokens)
            
            # Extract n-grams
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = ngram[:-1]
                next_token = ngram[-1]
                
                self.ngram_counts[context][next_token] += 1
                self.context_counts[context] += 1
                self.vocabulary.add(next_token)
        
        # Filter low-frequency n-grams
        if self.min_count > 1:
            self._filter_low_frequency()
        
        self.total_tokens = len(all_tokens)
        logger.info(f"Training complete. Vocabulary size: {len(self.vocabulary)}")
        logger.info(f"Unique contexts: {len(self.ngram_counts)}")
    
    def _filter_low_frequency(self):
        """Remove n-grams with counts below threshold."""
        filtered_ngrams = defaultdict(Counter)
        filtered_contexts = Counter()
        
        for context, next_tokens in self.ngram_counts.items():
            for token, count in next_tokens.items():
                if count >= self.min_count:
                    filtered_ngrams[context][token] = count
                    filtered_contexts[context] += count
        
        self.ngram_counts = filtered_ngrams
        self.context_counts = filtered_contexts
    
    def get_probability(self, context: Tuple[str, ...], next_token: str) -> float:
        """Get probability of next_token given context."""
        if len(context) != self.n - 1:
            raise ValueError(f"Context length must be {self.n - 1}")
        
        context_count = self.context_counts.get(context, 0)
        ngram_count = self.ngram_counts[context].get(next_token, 0)
        
        if self.smoothing == 'laplace':
            # Add-one smoothing
            return (ngram_count + 1) / (context_count + len(self.vocabulary))
        elif self.smoothing == 'none':
            if context_count == 0:
                return 1 / len(self.vocabulary) if self.vocabulary else 0
            return ngram_count / context_count
        else:
            # Default to Laplace
            return (ngram_count + 1) / (context_count + len(self.vocabulary))
    
    def predict_next_tokens(self, context: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Predict top-k next tokens with probabilities."""
        if len(context) < self.n - 1:
            # Pad with start-of-sequence tokens
            context = [self.tokenizer.SPECIAL_TOKENS['<SOS>']] * (self.n - 1 - len(context)) + context
        elif len(context) > self.n - 1:
            # Take last n-1 tokens
            context = context[-(self.n - 1):]
        
        context_tuple = tuple(context)
        
        # Get all possible next tokens for this context
        candidates = []
        if context_tuple in self.ngram_counts:
            for token, count in self.ngram_counts[context_tuple].items():
                prob = self.get_probability(context_tuple, token)
                candidates.append((token, prob))
        
        # If no candidates found, use vocabulary
        if not candidates:
            for token in self.vocabulary:
                if token not in self.tokenizer.SPECIAL_TOKENS.values():
                    prob = self.get_probability(context_tuple, token)
                    if prob > 0:
                        candidates.append((token, prob))
        
        # Sort by probability and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def sample_completion(self, context: List[str], max_length: int = 50) -> str:
        """Sample a completion given initial context."""
        current_context = context.copy()
        completion = []
        
        for _ in range(max_length):
            predictions = self.predict_next_tokens(current_context[-self.n + 1:], top_k=10)
            
            if not predictions:
                break
            
            # Sample from top predictions
            tokens, probs = zip(*predictions)
            
            # Convert to numpy array for sampling
            probs = np.array(probs)
            probs = probs / probs.sum()  # Normalize
            
            # Sample token
            next_token = np.random.choice(tokens, p=probs)
            
            # Stop conditions
            if (next_token == self.tokenizer.SPECIAL_TOKENS['<EOS>'] or
                next_token in ['}', ';'] and len(completion) > 3):
                completion.append(next_token)
                break
            
            completion.append(next_token)
            current_context.append(next_token)
        
        return ' '.join(completion)
    
    def calculate_perplexity(self, test_corpus: List[str]) -> float:
        """Calculate perplexity on test corpus."""
        total_log_prob = 0
        total_tokens = 0
        
        for code in test_corpus:
            tokens = self.tokenizer.tokenize(code)
            if len(tokens) < self.n:
                continue
            
            tokens = [self.tokenizer.SPECIAL_TOKENS['<SOS>']] + tokens + [self.tokenizer.SPECIAL_TOKENS['<EOS>']]
            
            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1:i])
                next_token = tokens[i]
                
                prob = self.get_probability(context, next_token)
                if prob > 0:
                    total_log_prob += math.log(prob)
                    total_tokens += 1
        
        if total_tokens == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)
        return perplexity

class NGramRecommender:
    """Main class for N-gram code recommendation system."""
    
    def __init__(self, csv_file: str):
        """Initialize recommender with dataset."""
        self.csv_file = csv_file
        self.df = None
        self.models = {}  # n -> model
        self.best_model = None
        self.best_n = None
    
    def load_data(self):
        """Load and preprocess the dataset."""
        logger.info(f"Loading dataset from {self.csv_file}")
        
        try:
            self.df = pd.read_csv(self.csv_file)
            logger.info(f"Loaded {len(self.df)} examples")
            
            # Debug: Print column names and first few rows
            logger.info(f"Available columns: {list(self.df.columns)}")
            
            # Check if dataset_split column exists, if not create it
            if 'dataset_split' not in self.df.columns:
                logger.warning("No 'dataset_split' column found. Creating train/eval split...")
                # Create 80/20 split
                split_idx = int(0.8 * len(self.df))
                self.df['dataset_split'] = ['train'] * split_idx + ['eval'] * (len(self.df) - split_idx)
            
            # Check for code column - try multiple possibilities
            code_column = None
            possible_code_cols = ['original_code', 'code_tokens', 'code', 'method_code', 'source_code']
            
            for col in possible_code_cols:
                if col in self.df.columns:
                    code_column = col
                    break
            
            if code_column is None:
                logger.error("No code column found. Available columns: " + str(list(self.df.columns)))
                raise ValueError("Could not find a suitable code column")
            
            # Standardize to 'original_code'
            if code_column != 'original_code':
                self.df['original_code'] = self.df[code_column]
                
            # Print split statistics
            train_count = len(self.df[self.df['dataset_split'] == 'train'])
            eval_count = len(self.df[self.df['dataset_split'] == 'eval'])
            logger.info(f"Train examples: {train_count}, Eval examples: {eval_count}")
            
            # Show a sample of data
            logger.info("Sample data:")
            sample_code = self.df['original_code'].dropna().iloc[0] if not self.df['original_code'].dropna().empty else "No code found"
            logger.info(f"First code sample: {sample_code[:100]}...")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def train_models(self, n_values: List[int] = [3, 5, 7]):
        """Train N-gram models with different N values."""
        if self.df is None:
            self.load_data()
        
        # Get training data
        train_data = self.df[self.df['dataset_split'] == 'train']['original_code'].dropna().tolist()
        logger.info(f"Training on {len(train_data)} examples")
        
        # Train models for different N values
        for n in n_values:
            logger.info(f"Training {n}-gram model...")
            model = NGramModel(n=n, smoothing='laplace', min_count=2)
            model.train(train_data)
            self.models[n] = model
    
    def evaluate_models(self):
        """Evaluate all trained models and select the best one."""
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        # Get evaluation data
        eval_data = self.df[self.df['dataset_split'] == 'eval']['original_code'].dropna().tolist()
        
        # If no eval data, use a subset of train data for evaluation
        if not eval_data:
            logger.warning("No evaluation data found. Using subset of training data for evaluation.")
            train_data = self.df[self.df['dataset_split'] == 'train']['original_code'].dropna().tolist()
            eval_data = train_data[-min(1000, len(train_data)//5):]  # Use last 20% or 1000 samples
        else:
            eval_data = eval_data[:1000]  # Limit for speed
        
        logger.info(f"Evaluating on {len(eval_data)} examples")
        
        best_perplexity = float('inf')
        
        for n, model in self.models.items():
            logger.info(f"Evaluating {n}-gram model...")
            try:
                perplexity = model.calculate_perplexity(eval_data)
                logger.info(f"{n}-gram perplexity: {perplexity:.2f}")
                
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    self.best_model = model
                    self.best_n = n
            except Exception as e:
                logger.warning(f"Error evaluating {n}-gram model: {e}")
                continue
        
        # If no valid model found, use the largest N model
        if self.best_model is None:
            logger.warning("No model could be evaluated properly. Using largest N model.")
            self.best_n = max(self.models.keys())
            self.best_model = self.models[self.best_n]
            best_perplexity = float('inf')
        
        logger.info(f"Best model: {self.best_n}-gram (perplexity: {best_perplexity:.2f})")
    
    def sample_predictions(self, num_samples: int = 1000, output_file: str = 'ngram_predictions.jsonl'):
        """Generate predictions on test samples."""
        if self.best_model is None:
            raise ValueError("No best model selected. Call evaluate_models() first.")
        
        # Get test data - try eval first, then fall back to train
        test_data = self.df[self.df['dataset_split'] == 'eval']['original_code'].dropna().tolist()
        if not test_data:
            logger.warning("No eval data found. Using training data for sampling.")
            test_data = self.df[self.df['dataset_split'] == 'train']['original_code'].dropna().tolist()
        
        # Sample from available data
        test_samples = random.sample(test_data, min(num_samples, len(test_data)))
        
        predictions = []
        tokenizer = self.best_model.tokenizer
        
        logger.info(f"Generating predictions for {len(test_samples)} samples...")
        
        successful_predictions = 0
        
        for i, code in enumerate(test_samples):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(test_samples)} samples")
            
            try:
                # Tokenize the code
                tokens = tokenizer.tokenize(code)
                if len(tokens) < self.best_n:
                    continue
                
                # Take first few tokens as context
                context_length = min(self.best_n - 1, max(1, len(tokens) // 3))
                context = tokens[:context_length]
                
                # Get predictions
                predictions_with_probs = self.best_model.predict_next_tokens(context, top_k=5)
                
                if not predictions_with_probs:  # Skip if no predictions
                    continue
                
                # Sample completion
                completion = self.best_model.sample_completion(context, max_length=20)
                
                # Store result
                result = {
                    'sample_id': successful_predictions,
                    'input_context': ' '.join(context),
                    'predictions': [{'token': token, 'probability': prob} for token, prob in predictions_with_probs],
                    'sampled_completion': completion,
                    'original_code_snippet': ' '.join(tokens[:context_length + 10])  # Show a bit more for reference
                }
                predictions.append(result)
                successful_predictions += 1
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        # Save predictions
        with open(output_file, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
        
        logger.info(f"Saved {len(predictions)} predictions to {output_file}")
        
        # Print some examples
        if predictions:
            logger.info("Example predictions:")
            for i, pred in enumerate(predictions[:5]):
                print(f"\nExample {i + 1}:")
                print(f"Input context: {pred['input_context']}")
                print("Predictions:")
                for p in pred['predictions']:
                    print(f"  '{p['token']}' → {p['probability']:.3f}")
                print(f"Sampled completion: {pred['sampled_completion']}")
        else:
            logger.warning("No successful predictions generated!")
        
        return len(predictions)
    
    def run_complete_pipeline(self, n_values: List[int] = [3, 5, 7], num_samples: int = 1000):
        """Run the complete N-gram recommendation pipeline."""
        logger.info("Starting N-gram code recommender pipeline...")
        
        # Load data
        self.load_data()
        
        # Train models
        self.train_models(n_values)
        
        # Evaluate models
        self.evaluate_models()
        
        # Generate predictions
        successful_preds = self.sample_predictions(num_samples)
        
        if successful_preds > 0:
            logger.info("Pipeline completed successfully!")
        else:
            logger.warning("Pipeline completed with warnings - no successful predictions generated!")
        
        return successful_preds

def main():
    """Main function to run the N-gram recommender."""
    parser = argparse.ArgumentParser(description='N-gram Code Recommender')
    parser.add_argument('csv_file', help='Path to the CSV dataset file')
    parser.add_argument('--n-values', nargs='+', type=int, default=[3, 5, 7],
                        help='N-gram sizes to evaluate (default: 3 5 7)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of test samples for prediction (default: 1000)')
    parser.add_argument('--output', default='ngram_predictions.jsonl',
                        help='Output file for predictions (default: ngram_predictions.jsonl)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        logger.error(f"Dataset file not found: {args.csv_file}")
        return
    
    # Create and run recommender
    recommender = NGramRecommender(args.csv_file)
    successful_preds = recommender.run_complete_pipeline(args.n_values, args.samples)
    
    print("\n" + "="*60)
    print("N-GRAM CODE RECOMMENDER - SUMMARY")
    print("="*60)
    if recommender.best_model:
        print(f"Best model: {recommender.best_n}-gram")
        print(f"Vocabulary size: {len(recommender.best_model.vocabulary)}")
        print(f"Successful predictions: {successful_preds}")
        print(f"Predictions saved to: {args.output}")
    else:
        print("No successful model training completed")
    print("="*60)

if __name__ == "__main__":
    main()
