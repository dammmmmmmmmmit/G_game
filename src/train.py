"""
Training Script for Genome Language Model

Trains the GLM using Masked Language Modeling (MLM) on DNA sequences.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import DNATokenizer
from dataset import create_data_loaders
from model import GenomeLM

# Resolve project root (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Trainer:
    def __init__(self, model, train_loader, val_loader, tokenizer,
                 learning_rate=0.00001, weight_decay=0.01, num_epochs=10,
                 device=None, checkpoint_dir=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Training on: {self.device}")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir or os.path.join(PROJECT_ROOT, "models", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay, betas=(0.9, 0.999))
        total_steps = len(train_loader) * num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, n = 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        for batch in pbar:
            ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(ids, mask, labels)['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item(); n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.6f}")
        return total_loss / n

    def validate(self):
        self.model.eval()
        total_loss, n = 0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                loss = self.model(ids, mask, labels)['loss']
                total_loss += loss.item(); n += 1
        return total_loss / n

    def train(self):
        print("=" * 60 + f"\nSTARTING TRAINING ({self.num_epochs} epochs)\n" + "=" * 60)
        for epoch in range(self.num_epochs):
            tl = self.train_epoch(epoch)
            self.history['train_loss'].append(tl)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            vl = self.validate()
            self.history['val_loss'].append(vl)
            print(f"Epoch {epoch+1}: train_loss={tl:.4f}  val_loss={vl:.4f}")
            if vl < self.best_val_loss:
                self.best_val_loss = vl
                self._save("best_model.pt"); print("  [OK] Best model saved!")
            if (epoch + 1) % 5 == 0:
                self._save(f"checkpoint_epoch_{epoch+1}.pt")
        self._save("final_model.pt")
        self._plot()
        print("TRAINING COMPLETE!")

    def _save(self, name):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
        }, path)

    def _plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history['train_loss'], 'o-', label='Train')
        axes[0].plot(self.history['val_loss'], 's-', label='Val')
        axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(self.history['learning_rate'], 'go-')
        axes[1].set(xlabel='Epoch', ylabel='LR', title='Learning Rate'); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, "training_history.png"), dpi=150)
        plt.close()
        with open(os.path.join(self.checkpoint_dir, "training_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    print("=" * 60 + "\nBRAF MUTATION DETECTION - MODEL TRAINING\n" + "=" * 60)
    print(f"Time: {datetime.now()}")
    cfg = dict(k=3, vocab_size=None, d_model=256, num_heads=8, num_layers=6,
               d_ff=1024, max_length=512, dropout=0.1, batch_size=32,
               learning_rate=1e-4, weight_decay=0.01, num_epochs=50)
    print("Config:", json.dumps(cfg, indent=2))

    tokenizer = DNATokenizer(k=cfg['k'])
    cfg['vocab_size'] = tokenizer.vocab_size  # Update config with actual vocab size
    
    train_loader, val_loader = create_data_loaders(
        tokenizer, batch_size=cfg['batch_size'], max_length=cfg['max_length'])
    model = GenomeLM(vocab_size=cfg['vocab_size'], d_model=cfg['d_model'],
                     num_heads=cfg['num_heads'], num_layers=cfg['num_layers'],
                     d_ff=cfg['d_ff'], max_length=cfg['max_length'], dropout=cfg['dropout'])

    trainer = Trainer(model, train_loader, val_loader, tokenizer,
                      learning_rate=cfg['learning_rate'],
                      weight_decay=cfg['weight_decay'],
                      num_epochs=cfg['num_epochs'])

    tok_path = os.path.join(PROJECT_ROOT, "models", "tokenizer_config.json")
    tokenizer.save(tok_path)
    cfg_path = os.path.join(PROJECT_ROOT, "models", "config.json")
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    trainer.train()
    print(f"Done: {datetime.now()}")


if __name__ == "__main__":
    main()
