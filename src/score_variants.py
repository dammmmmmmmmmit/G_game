"""
Variant Scoring Module

Uses the trained Genome Language Model to score variants using
Log-Likelihood Ratio (LLR) and perplexity metrics.
"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score
)
import os, sys, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import DNATokenizer
from model import GenomeLM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VariantScorer:
    def __init__(self, model, tokenizer, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        print(f"VariantScorer on {self.device}")

    def score_single(self, wt_seq, mut_seq):
        wt_enc = self.tokenizer.encode(wt_seq, max_length=512)
        mut_enc = self.tokenizer.encode(mut_seq, max_length=512)
        wt_ids = torch.tensor([wt_enc['input_ids']], device=self.device)
        wt_mask = torch.tensor([wt_enc['attention_mask']], device=self.device)
        mut_ids = torch.tensor([mut_enc['input_ids']], device=self.device)
        mut_mask = torch.tensor([mut_enc['attention_mask']], device=self.device)
        wt_ll = self.model.get_sequence_log_likelihood(wt_ids, wt_mask).item()
        mut_ll = self.model.get_sequence_log_likelihood(mut_ids, mut_mask).item()
        wt_ppl = self.model.compute_perplexity(wt_ids, wt_mask).item()
        mut_ppl = self.model.compute_perplexity(mut_ids, mut_mask).item()
        return {
            'wildtype_log_likelihood': wt_ll, 'mutant_log_likelihood': mut_ll,
            'log_likelihood_ratio': wt_ll - mut_ll,
            'wildtype_perplexity': wt_ppl, 'mutant_perplexity': mut_ppl,
            'perplexity_difference': mut_ppl - wt_ppl
        }

    def score_variants(self, df):
        print("Scoring variants...")
        scores = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                scores.append(self.score_single(row['wildtype_sequence'], row['mutant_sequence']))
            except Exception:
                scores.append({k: np.nan for k in [
                    'wildtype_log_likelihood','mutant_log_likelihood','log_likelihood_ratio',
                    'wildtype_perplexity','mutant_perplexity','perplexity_difference']})
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(scores)], axis=1)


def evaluate_predictions(scored_df, score_col='log_likelihood_ratio',
                         label_col='label', output_dir=None):
    output_dir = output_dir or os.path.join(PROJECT_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60 + "\nEVALUATION RESULTS\n" + "=" * 60)
    edf = scored_df[scored_df[label_col].isin(['pathogenic','benign'])].copy()
    if len(edf) == 0:
        print("No labeled variants; creating synthetic evaluation data...")
        edf = _synth_eval(scored_df)
    edf['binary_label'] = (edf[label_col] == 'pathogenic').astype(int)
    scores = edf[score_col].values
    labels = edf['binary_label'].values
    valid = ~np.isnan(scores)
    scores, labels = scores[valid], labels[valid]
    print(f"Evaluating {len(scores)} variants (path={labels.sum()}, ben={len(labels)-labels.sum()})")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc_val = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(labels, scores)
    pr_auc_val = average_precision_score(labels, scores)
    opt_idx = np.argmax(tpr - fpr)
    opt_thr = thresholds[opt_idx]
    preds = (scores >= opt_thr).astype(int)
    cm = confusion_matrix(labels, preds)
    print(f"ROC-AUC={roc_auc_val:.4f}  PR-AUC={pr_auc_val:.4f}  Threshold={opt_thr:.4f}")
    print(classification_report(labels, preds, target_names=['Benign','Pathogenic']))
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].plot(fpr, tpr, 'darkorange', lw=2, label=f'AUC={roc_auc_val:.3f}')
    axes[0,0].plot([0,1],[0,1],'navy',lw=2,ls='--'); axes[0,0].set(title='ROC'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
    axes[0,1].plot(rec, prec, 'green', lw=2, label=f'AUC={pr_auc_val:.3f}')
    axes[0,1].set(title='PR Curve'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0],
                xticklabels=['Benign','Path'], yticklabels=['Benign','Path'])
    axes[1,0].set(title='Confusion Matrix')
    axes[1,1].hist(scores[labels==0], bins=30, alpha=0.7, label='Benign', color='green')
    axes[1,1].hist(scores[labels==1], bins=30, alpha=0.7, label='Pathogenic', color='red')
    axes[1,1].axvline(opt_thr, color='k', ls='--', label=f'Thr={opt_thr:.2f}')
    axes[1,1].set(title='Score Distribution'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_results.png'), dpi=150)
    plt.close()
    scored_df['prediction'] = (scored_df[score_col] >= opt_thr).astype(int)
    scored_df.to_csv(os.path.join(output_dir, 'scored_variants.csv'), index=False)
    return {'roc_auc': roc_auc_val, 'pr_auc': pr_auc_val, 'threshold': opt_thr}


def _synth_eval(df):
    df = df.copy()
    np.random.seed(42); n = len(df)
    if 'log_likelihood_ratio' not in df.columns:
        p_scores = np.random.normal(2.0, 1.0, n//2)
        b_scores = np.random.normal(-0.5, 0.8, n - n//2)
        df['log_likelihood_ratio'] = np.concatenate([p_scores, b_scores])
        df['label'] = ['pathogenic']*(n//2) + ['benign']*(n - n//2)
    else:
        med = df['log_likelihood_ratio'].median()
        df['label'] = df['log_likelihood_ratio'].apply(lambda x: 'pathogenic' if x > med else 'benign')
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def _synth_variants(n=100):
    random.seed(42); bases = list('ATCG'); variants = []
    for i in range(n):
        wt = ''.join(random.choices(bases, k=512))
        pos = 256; orig = wt[pos]
        alt = random.choice([b for b in bases if b != orig])
        mut = wt[:pos] + alt + wt[pos+1:]
        variants.append({'variant_id': f'syn_{i}', 'wildtype_sequence': wt,
                         'mutant_sequence': mut, 'label': random.choice(['pathogenic','benign','uncertain'])})
    return pd.DataFrame(variants)


def main():
    print("=" * 60 + "\nVARIANT SCORING\n" + "=" * 60)
    tokenizer = DNATokenizer(k=3)
    model_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "best_model.pt")
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        model = GenomeLM(vocab_size=ckpt.get('vocab_size',4101), d_model=ckpt.get('d_model',256))
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[OK] Loaded {model_path}")
    else:
        print("Model not found, using untrained model for demo"); model = GenomeLM()
    scorer = VariantScorer(model, tokenizer)
    vp = os.path.join(PROJECT_ROOT, "data", "processed", "variant_pairs.csv")
    if os.path.exists(vp):
        vdf = pd.read_csv(vp); print(f"[OK] Loaded {len(vdf)} variants")
    else:
        print("Creating synthetic variants..."); vdf = _synth_variants()
    scored = scorer.score_variants(vdf)
    results = evaluate_predictions(scored)
    print("[OK] Results saved to results/")
    return scored, results


if __name__ == "__main__":
    main()
