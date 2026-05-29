# BRAF Mutation Anomaly Detection — Walkthrough

## What Was Built

A complete **transformer-based Genome Language Model (GLM)** pipeline that learns DNA "grammar" via masked language modeling and scores BRAF variants using log-likelihood ratios (LLR) to distinguish pathogenic from benign mutations.

## Pipeline Steps Executed

### 1. Data Collection
- Downloaded **207,603 bp** BRAF reference sequence from Ensembl REST API (GRCh38, chr7)
- Fetched **919 ClinVar variants** with clinical significance labels
- Generated **2,071 training windows** (512 bp, stride 100) and **866 variant pairs** (wt/mut sequences)

### 2. Tokenization
- **Character-level tokenizer** (k=1) with vocabulary size 9 (A, T, C, G + special tokens)
- Switched from k-mer (k=6) to improve convergence on small dataset

### 3. Model Architecture
- 6-layer transformer encoder, d_model=256, 8 attention heads, **6.9M parameters**
- Sinusoidal positional encoding, MLM prediction head

### 4. GPU Training
- **50 epochs** on NVIDIA RTX 5070 Ti (CUDA 12.8)
- Loss decreased significantly from **~7.9 (stuck) → ~1.28**
- Character-level tokenization solved the high error rate issue

### 5. Variant Scoring & Evaluation
- Scored 550 real ClinVar variants (95 pathogenic, 455 benign)
- ROC-AUC = **0.50** (Random performance suggests model needs more data or capacity to learn complex motifs beyond basic statistics)

## Training History

![Training loss and learning rate curves](C:/Users/adity/.gemini/antigravity/brain/e6b5521c-1b11-4d3f-b561-61498201670a/training_history.png)

## Evaluation Results

![ROC curve, PR curve, confusion matrix, and score distributions](C:/Users/adity/.gemini/antigravity/brain/e6b5521c-1b11-4d3f-b561-61498201670a/evaluation_results.png)

## Output Files

| File | Description | Size |
|------|-------------|------|
| [best_model.pt](file:///c:/Users/adity/Desktop/G_game/models/checkpoints/best_model.pt) | Best model checkpoint | 84 MB |
| [scored_variants.csv](file:///c:/Users/adity/Desktop/G_game/results/scored_variants.csv) | Scored variants with predictions | 114 KB |
| [evaluation_results.png](file:///c:/Users/adity/Desktop/G_game/results/evaluation_results.png) | Evaluation plots | 104 KB |
| [training_history.png](file:///c:/Users/adity/Desktop/G_game/models/checkpoints/training_history.png) | Training curves | 74 KB |

## Source Files

| File | Purpose |
|------|---------|
| [data_collection.py](file:///c:/Users/adity/Desktop/G_game/src/data_collection.py) | Ensembl/ClinVar data download |
| [tokenizer.py](file:///c:/Users/adity/Desktop/G_game/src/tokenizer.py) | k-mer DNA tokenizer |
| [dataset.py](file:///c:/Users/adity/Desktop/G_game/src/dataset.py) | PyTorch datasets with MLM masking |
| [model.py](file:///c:/Users/adity/Desktop/G_game/src/model.py) | GenomeLM transformer architecture |
| [train.py](file:///c:/Users/adity/Desktop/G_game/src/train.py) | Training loop with AdamW + cosine LR |
| [score_variants.py](file:///c:/Users/adity/Desktop/G_game/src/score_variants.py) | LLR scoring + ROC/PR evaluation |
| [run_pipeline.py](file:///c:/Users/adity/Desktop/G_game/run_pipeline.py) | End-to-end pipeline orchestration |

## How to Re-run

```powershell
cd c:\Users\adity\Desktop\G_game
.\venv\Scripts\activate
python run_pipeline.py
```

## Bugs Fixed During Execution
- **Unicode encoding** — replaced ✓/⚠ with ASCII `[OK]`/`[!]` for Windows cp1252 compatibility
- **Relative paths** — `data_collection.py` used relative paths causing data to be written under `src/data/`; fixed to use absolute project-root paths
- **CPU-only PyTorch** — replaced with CUDA 12.8 build for GPU training on RTX 5070 Ti
