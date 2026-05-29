"""
Genome Language Model (GLM) Architecture

Lightweight transformer-based language model for DNA sequences,
designed for masked language modeling and variant effect prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict
import os


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, S, _ = x.shape
        q = self.w_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.w_o(context)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        x = self.norm1(x + self.dropout(self.attention(x, attention_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class GenomeLM(nn.Module):
    def __init__(self, vocab_size=4101, d_model=256, num_heads=8,
                 num_layers=6, d_ff=1024, max_length=512,
                 dropout=0.1, pad_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(d_model, max_length, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)])
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size))
        self._init_weights()
        total = sum(p.numel() for p in self.parameters())
        print(f"Model: {total:,} parameters")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.positional_encoding(self.embedding(input_ids))
        for layer in self.layers:
            x = layer(x, attention_mask)
        logits = self.mlm_head(x)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.vocab_size), labels.view(-1))
        return {'logits': logits, 'loss': loss, 'hidden_states': x}

    def get_sequence_log_likelihood(self, input_ids, attention_mask=None):
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)['logits']
            log_probs = F.log_softmax(logits, dim=-1)
            token_lp = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
            if attention_mask is not None:
                token_lp = token_lp * attention_mask.float()
            return token_lp.sum(dim=-1)

    def compute_perplexity(self, input_ids, attention_mask=None):
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)['logits']
            log_probs = F.log_softmax(logits, dim=-1)
            token_lp = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
            if attention_mask is not None:
                token_lp = token_lp * attention_mask.float()
                lengths = attention_mask.sum(-1).float()
                mean_lp = token_lp.sum(-1) / lengths
            else:
                mean_lp = token_lp.mean(-1)
            return torch.exp(-mean_lp)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({'model_state_dict': self.state_dict(),
                     'vocab_size': self.vocab_size, 'd_model': self.d_model}, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, device='cpu'):
        ckpt = torch.load(path, map_location=device)
        model = cls(vocab_size=ckpt['vocab_size'], d_model=ckpt['d_model'])
        model.load_state_dict(ckpt['model_state_dict'])
        return model


def test_model():
    print("=" * 60 + "\nMODEL TEST\n" + "=" * 60)
    vocab_size, bs, sl = 4101, 4, 128
    model = GenomeLM(vocab_size=vocab_size, d_model=256, num_heads=8,
                     num_layers=6, d_ff=1024, max_length=512, dropout=0.1)
    ids = torch.randint(0, vocab_size, (bs, sl))
    mask = torch.ones(bs, sl)
    labels = ids.clone(); labels[labels < 5] = -100
    out = model(ids, mask, labels)
    print(f"Logits: {out['logits'].shape}, Loss: {out['loss'].item():.4f}")
    ll = model.get_sequence_log_likelihood(ids, mask)
    print(f"Log-likelihood: {ll}")
    ppl = model.compute_perplexity(ids, mask)
    print(f"Perplexity: {ppl}")
    os.makedirs("models", exist_ok=True)
    model.save("models/test_model.pt")
    GenomeLM.load("models/test_model.pt")
    print("[OK] Model test complete!")


if __name__ == "__main__":
    test_model()
