"""
K-mer Tokenizer for DNA Sequences

This module implements a k-mer tokenizer that converts DNA sequences
into tokens that can be processed by a transformer model.

K-mer tokenization splits DNA into overlapping subsequences of length k.
Example with k=6:
    "ATCGATCG" -> ["ATCGAT", "TCGATC", "CGATCG"]
"""

import os
import json
from typing import List, Dict, Tuple
from collections import Counter
import itertools


class DNATokenizer:
    """
    DNA Tokenizer using k-mer approach.
    
    Special tokens:
    - [PAD]: Padding token (index 0)
    - [UNK]: Unknown token (index 1)
    - [CLS]: Classification token (index 2)
    - [SEP]: Separator token (index 3)
    - [MASK]: Mask token for MLM (index 4)
    """
    
    def __init__(self, k: int = 6):
        """
        Initialize the tokenizer.
        
        Args:
            k: Size of k-mers (default 6, giving 4^6 = 4096 possible k-mers)
        """
        self.k = k
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        
        # Build vocabulary
        self.vocab = self._build_vocabulary()
        self.vocab_size = len(self.vocab)
        
        # Reverse vocabulary for decoding
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"Initialized DNATokenizer with k={k}")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """
        Build vocabulary of all possible k-mers.
        
        For k=6, this creates 4^6 = 4096 k-mers plus special tokens.
        """
        vocab = dict(self.special_tokens)  # Start with special tokens
        
        # Generate all possible k-mers
        bases = ['A', 'T', 'C', 'G']
        
        idx = len(self.special_tokens)
        for kmer in itertools.product(bases, repeat=self.k):
            kmer_str = ''.join(kmer)
            vocab[kmer_str] = idx
            idx += 1
        
        return vocab
    
    def tokenize(self, sequence: str) -> List[str]:
        """
        Convert DNA sequence to list of k-mer tokens.
        
        Args:
            sequence: DNA sequence string (A, T, C, G)
            
        Returns:
            List of k-mer strings
        """
        sequence = sequence.upper().strip()
        
        tokens = []
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            tokens.append(kmer)
        
        return tokens
    
    def encode(self, sequence: str, 
               max_length: int = 512,
               add_special_tokens: bool = True,
               padding: bool = True,
               truncation: bool = True) -> Dict[str, List[int]]:
        """
        Encode DNA sequence to token IDs.
        
        Args:
            sequence: DNA sequence string
            max_length: Maximum sequence length
            add_special_tokens: Whether to add [CLS] and [SEP]
            padding: Whether to pad to max_length
            truncation: Whether to truncate if too long
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Tokenize
        tokens = self.tokenize(sequence)
        
        # Convert to IDs
        input_ids = []
        
        if add_special_tokens:
            input_ids.append(self.special_tokens['[CLS]'])
        
        for token in tokens:
            if token in self.vocab:
                input_ids.append(self.vocab[token])
            else:
                # Handle unknown k-mers (containing N or other characters)
                input_ids.append(self.special_tokens['[UNK]'])
        
        if add_special_tokens:
            input_ids.append(self.special_tokens['[SEP]'])
        
        # Truncation
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            if add_special_tokens:
                input_ids[-1] = self.special_tokens['[SEP]']
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Padding
        if padding and len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids.extend([self.special_tokens['[PAD]']] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to DNA sequence.
        
        Note: This won't perfectly reconstruct the original due to
        overlapping k-mers, but useful for debugging.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Reconstructed sequence (approximate)
        """
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                if token not in self.special_tokens:
                    tokens.append(token)
        
        if not tokens:
            return ""
        
        # Reconstruct: first k-mer fully, then last character of each subsequent
        sequence = tokens[0]
        for token in tokens[1:]:
            sequence += token[-1]
        
        return sequence
    
    def batch_encode(self, sequences: List[str], 
                     max_length: int = 512,
                     add_special_tokens: bool = True,
                     padding: bool = True,
                     truncation: bool = True) -> Dict[str, List[List[int]]]:
        """
        Encode multiple sequences.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Dictionary with batched 'input_ids' and 'attention_mask'
        """
        all_input_ids = []
        all_attention_masks = []
        
        for seq in sequences:
            encoded = self.encode(
                seq,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation
            )
            all_input_ids.append(encoded['input_ids'])
            all_attention_masks.append(encoded['attention_mask'])
        
        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks
        }
    
    def save(self, path: str):
        """Save tokenizer configuration to file."""
        config = {
            'k': self.k,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DNATokenizer':
        """Load tokenizer from configuration file."""
        with open(path, 'r') as f:
            config = json.load(f)
        
        return cls(k=config['k'])
    
    def get_mask_token_id(self) -> int:
        """Get the ID of the [MASK] token."""
        return self.special_tokens['[MASK]']
    
    def get_pad_token_id(self) -> int:
        """Get the ID of the [PAD] token."""
        return self.special_tokens['[PAD]']


def test_tokenizer():
    """Test the tokenizer with sample sequences."""
    
    print("=" * 60)
    print("TOKENIZER TEST")
    print("=" * 60)
    
    # Initialize tokenizer
    tokenizer = DNATokenizer(k=6)
    
    # Test sequence
    test_seq = "ATCGATCGATCGATCGATCGATCGATCG"
    
    print(f"\nTest sequence: {test_seq}")
    print(f"Length: {len(test_seq)}")
    
    # Tokenize
    tokens = tokenizer.tokenize(test_seq)
    print(f"\nTokens: {tokens[:5]}... (showing first 5)")
    print(f"Number of tokens: {len(tokens)}")
    
    # Encode
    encoded = tokenizer.encode(test_seq, max_length=32)
    print(f"\nEncoded input_ids: {encoded['input_ids'][:10]}... (showing first 10)")
    print(f"Attention mask: {encoded['attention_mask'][:10]}...")
    
    # Decode
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"\nDecoded sequence: {decoded}")
    
    # Save tokenizer
    tokenizer.save("models/tokenizer_config.json")
    
    print("\n[OK] Tokenizer test complete!")


if __name__ == "__main__":
    test_tokenizer()
