"""
PyTorch Dataset for DNA Sequences

This module creates datasets for training the genome language model
using masked language modeling (MLM).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
import sys
import os

# Add src directory to path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import DNATokenizer


class DNADataset(Dataset):
    """
    Dataset for DNA sequences with Masked Language Modeling.
    
    During training, random tokens are masked and the model
    learns to predict them from context.
    """
    
    def __init__(self, 
                 sequences: List[str],
                 tokenizer: DNATokenizer,
                 max_length: int = 512,
                 mlm_probability: float = 0.15):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of DNA sequences
            tokenizer: DNATokenizer instance
            max_length: Maximum sequence length
            mlm_probability: Fraction of tokens to mask (default 15%)
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
        # Pre-tokenize all sequences for efficiency
        print("Pre-tokenizing sequences...")
        self.encoded_sequences = []
        
        for seq in tqdm(sequences):
            encoded = tokenizer.encode(
                seq,
                max_length=max_length,
                add_special_tokens=True,
                padding=True,
                truncation=True
            )
            self.encoded_sequences.append(encoded)
        
        print(f"Dataset created with {len(self.encoded_sequences)} sequences")
    
    def __len__(self) -> int:
        return len(self.encoded_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item with MLM masking applied.
        
        Returns:
            Dictionary containing:
            - input_ids: Token IDs with some masked
            - attention_mask: Attention mask
            - labels: Original token IDs (for computing loss)
        """
        encoded = self.encoded_sequences[idx]
        
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)
        
        # Create labels (copy of original input_ids)
        labels = input_ids.clone()
        
        # Apply MLM masking
        input_ids, labels = self._apply_mlm_masking(input_ids, labels, attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _apply_mlm_masking(self, 
                           input_ids: torch.Tensor, 
                           labels: torch.Tensor,
                           attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MLM masking strategy:
        - 15% of tokens are selected for prediction
        - Of those: 80% replaced with [MASK], 10% random token, 10% unchanged
        
        Args:
            input_ids: Original token IDs
            labels: Label tensor (will be modified)
            attention_mask: Attention mask
            
        Returns:
            Tuple of (masked_input_ids, labels)
        """
        # Get special token IDs
        mask_token_id = self.tokenizer.get_mask_token_id()
        pad_token_id = self.tokenizer.get_pad_token_id()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        
        # Don't mask special tokens (first 5 in vocab)
        special_tokens_mask = input_ids < 5
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Don't mask padding
        padding_mask = attention_mask == 0
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # Sample tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels to -100 for non-masked tokens (ignored in loss)
        labels[~masked_indices] = -100
        
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 0.8)
        ).bool() & masked_indices
        input_ids[indices_replaced] = mask_token_id
        
        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(
            5, self.tokenizer.vocab_size,  # Avoid special tokens
            input_ids.shape,
            dtype=torch.long
        )
        input_ids[indices_random] = random_tokens[indices_random]
        
        # 10% of the time, keep original (already done by not modifying)
        
        return input_ids, labels


class VariantDataset(Dataset):
    """
    Dataset for variant scoring (not for training).
    
    Contains pairs of wild-type and mutant sequences for
    computing log-likelihood ratios.
    """
    
    def __init__(self,
                 variant_pairs_df: pd.DataFrame,
                 tokenizer: DNATokenizer,
                 max_length: int = 512):
        """
        Initialize variant dataset.
        
        Args:
            variant_pairs_df: DataFrame with 'wildtype_sequence' and 'mutant_sequence'
            tokenizer: DNATokenizer instance
            max_length: Maximum sequence length
        """
        self.df = variant_pairs_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-encode sequences
        print("Pre-encoding variant pairs...")
        self.encoded_pairs = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            wt_encoded = tokenizer.encode(
                row['wildtype_sequence'],
                max_length=max_length,
                add_special_tokens=True,
                padding=True,
                truncation=True
            )
            
            mut_encoded = tokenizer.encode(
                row['mutant_sequence'],
                max_length=max_length,
                add_special_tokens=True,
                padding=True,
                truncation=True
            )
            
            self.encoded_pairs.append({
                'wildtype': wt_encoded,
                'mutant': mut_encoded,
                'label': row.get('label', 'unknown'),
                'variant_id': row.get('variant_id', f'var_{idx}')
            })
        
        print(f"Variant dataset created with {len(self.encoded_pairs)} pairs")
    
    def __len__(self) -> int:
        return len(self.encoded_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        pair = self.encoded_pairs[idx]
        
        return {
            'wildtype_input_ids': torch.tensor(pair['wildtype']['input_ids'], dtype=torch.long),
            'wildtype_attention_mask': torch.tensor(pair['wildtype']['attention_mask'], dtype=torch.long),
            'mutant_input_ids': torch.tensor(pair['mutant']['input_ids'], dtype=torch.long),
            'mutant_attention_mask': torch.tensor(pair['mutant']['attention_mask'], dtype=torch.long),
            'label': pair['label'],
            'variant_id': pair['variant_id']
        }


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
                  'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
    return ''.join(complement.get(base, base) for base in reversed(seq))


def create_data_loaders(tokenizer: DNATokenizer,
                        batch_size: int = 32,
                        max_length: int = 512,
                        train_split: float = 0.8,
                        augment: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        tokenizer: DNATokenizer instance
        batch_size: Batch size for training
        max_length: Maximum sequence length
        train_split: Fraction of data for training
        augment: Whether to add reverse complement augmentation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    print("=" * 60)
    print("CREATING DATA LOADERS")
    print("=" * 60)
    
    # Resolve the project root (one level above src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    training_data_path = os.path.join(project_root, "data", "processed", "training_windows.csv")
    
    # Load training windows
    try:
        df = pd.read_csv(training_data_path)
        sequences = df['sequence'].tolist()
        print(f"Loaded {len(sequences)} sequences from training_windows.csv")
    except FileNotFoundError:
        print("Training data not found. Creating sample data...")
        sequences = create_sample_sequences(1000)
    
    # Data augmentation: add reverse complements
    if augment:
        rc_sequences = [reverse_complement(seq) for seq in sequences]
        sequences = sequences + rc_sequences
        print(f"After reverse complement augmentation: {len(sequences)} sequences")
    
    # Shuffle and split
    random.shuffle(sequences)
    split_idx = int(len(sequences) * train_split)
    
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    # Create datasets
    train_dataset = DNADataset(
        train_sequences,
        tokenizer,
        max_length=max_length
    )
    
    val_dataset = DNADataset(
        val_sequences,
        tokenizer,
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def create_sample_sequences(num_sequences: int = 1000) -> List[str]:
    """Create sample DNA sequences for testing."""
    
    bases = ['A', 'T', 'C', 'G']
    sequences = []
    
    for _ in range(num_sequences):
        length = random.randint(400, 512)
        seq = ''.join(random.choices(bases, k=length))
        sequences.append(seq)
    
    return sequences


def test_dataset():
    """Test the dataset creation."""
    
    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)
    
    # Create tokenizer
    tokenizer = DNATokenizer(k=6)
    
    # Create sample sequences
    sequences = create_sample_sequences(100)
    
    # Create dataset
    dataset = DNADataset(sequences, tokenizer, max_length=128)
    
    # Get a sample
    sample = dataset[0]
    
    print(f"\nSample batch:")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
    
    # Count masked tokens
    masked_count = (sample['input_ids'] == tokenizer.get_mask_token_id()).sum().item()
    total_tokens = (sample['attention_mask'] == 1).sum().item()
    print(f"  Masked tokens: {masked_count}/{total_tokens}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        tokenizer,
        batch_size=16,
        max_length=128
    )
    
    # Test iteration
    for batch in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Input IDs: {batch['input_ids'].shape}")
        print(f"  Attention mask: {batch['attention_mask'].shape}")
        print(f"  Labels: {batch['labels'].shape}")
        break
    
    print("\n[OK] Dataset test complete!")


if __name__ == "__main__":
    test_dataset()
