# BRAF Gene Mutation Anomaly Detection Using LLR and Sequence Likelihood Modeling

[![Paper Draft](https://img.shields.io/badge/Academic--Paper-Google--Doc-blue?style=for-the-badge&logo=google-docs&logoColor=white)](https://docs.google.com/document/d/1bTAlS9ckqM_CJiNlqLh-MRAG_Z0oRLKWe54ornZ0TEI/edit?usp=sharing)
[![Model Platform](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Project--Status-Active-brightgreen?style=for-the-badge)](https://github.com/dammmmmmmmmmit/G_game)

---

## 📌 Project Overview
Somatic mutations in melanoma present a complex genomic landscape characterized by a exceptionally high mutational burden (primarily driven by UV-induced DNA damage). Distinguishing pathogenic **driver mutations** from neutral **passenger mutations** is a cornerstone of precision oncology. 

This repository hosts a complete, lightweight **Genome Language Model (GLM)** pipeline designed for the **unsupervised anomaly detection of melanoma-associated mutations** in the human **BRAF gene**. Utilizing a Transformer Encoder architecture trained via Self-Supervised Masked Language Modeling (MLM), the model learns the statistical "grammar" of DNA directly from nucleotide sequences without relying on clinical labels. Variant pathogenicity is subsequently quantified through a probabilistic **Log-Likelihood Ratio (LLR)** scoring framework that measures mutational "surprisal".

> **Research Paper:**
> This repository implements the methodology, training, and evaluation frameworks presented in:
> * **"BRAF Gene Mutation Anomaly Detection Using LLR and Sequence Likelihood Modeling"**
> * **Author:** Aditya Sharma (PG Scholar, Department of Computer Science, Christ University, Bengaluru)
> * **Paper Link:** [Read the full paper on Google Docs](https://docs.google.com/document/d/1bTAlS9ckqM_CJiNlqLh-MRAG_Z0oRLKWe54ornZ0TEI/edit?usp=sharing)

---

## 🔬 Core Methodology & Science

```
                          ┌───────────────────────────┐
                          │    GRCh38 BRAF Region     │
                          └─────────────┬─────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │   512 bp Sliding Windows  │
                          └─────────────┬─────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │ Character-Level Tokenizer │
                          └─────────────┬─────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │  Masked Language Model    │
                          │   (6-Layer Transformer)   │
                          └─────────────┬─────────────┘
                                        ▼
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
      ┌─────────────────────────────┐       ┌─────────────────────────────┐
      │     Wild-Type Sequence      │       │       Mutant Sequence       │
      │    log P(wild-type)         │       │     log P(mutant)           │
      └──────────────┬──────────────┘       └──────────────┬──────────────┘
                     └──────────────────┬──────────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │   Log-Likelihood Ratio    │
                          │     LLR Anomaly Score     │
                          └───────────────────────────┘
```

### 1. Self-Supervised Genome Grammar Learning
The model treats the DNA sequence as a natural language. During the training phase, the model is fed wild-type genomic sequence windows from the **GRCh38 (Chromosome 7)** region corresponding to the BRAF gene. 
* **Tokenization:** A character-level tokenizer ($k=1$) is utilized, mapping DNA nucleotides (A, T, C, G) along with special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`) directly to a 9-token vocabulary. This single-nucleotide resolution captures precise mutation sites without the massive vocabulary overhead of $k$-mer formulations.
* **Pre-training Objective:** Masked Language Modeling (MLM). Randomly selected tokens (15% probability) are replaced with `[MASK]`. The model is trained using Cross-Entropy Loss to predict the original nucleotide in the masked positions by analyzing the surrounding context.

### 2. Variant Pathogenicity Scoring (Log-Likelihood Ratio)
To evaluate the impact of a genomic variant, the model compares the likelihood of the **Wild-Type (WT)** sequence against the **Mutant (MUT)** sequence. The variant "surprisal" is quantified through the **Log-Likelihood Ratio (LLR)**:

$$\text{LLR} = \log P(\text{Wild-Type}) - \log P(\text{Mutant})$$

Where $\log P(\mathbf{S})$ is computed by summing the token-wise log-probabilities predicted by the model over the sequence windows centered on the variant position:

$$\log P(\mathbf{S}) = \sum_{i \in \text{window}} \log P(s_i \mid \mathbf{S}_{\backslash i})$$

* **High Positive LLR ($> +1.0$):** Indicates the mutant sequence is significantly *less natural* (lower probability) compared to the wild-type, signaling a high likelihood of functional disruption or **pathogenicity**.
* **Zero or Negative LLR ($\le 0$):** Indicates the mutation is highly consistent with the statistical grammar learned by the model, signaling a **benign** or neutral variant.

---

## 🛠️ Model Architecture

The custom Genome Language Model is implemented in PyTorch with the following specifications:

| Parameter | Configuration Value |
| :--- | :--- |
| **Model Type** | Transformer Encoder |
| **Attention Layers** | 6 Encoder Layers |
| **Attention Heads** | 8 Heads |
| **Embedding Dimension ($d_{model}$)** | 256 |
| **Feed-forward Dimension ($d_{ff}$)** | 1024 |
| **Context Window Size** | 512 bp |
| **Total Trainable Parameters** | ~6.9 Million |
| **Positional Encoding** | Sinusoidal Positional Encoding |

---

## 🚀 End-to-End Pipeline Steps

### 1. Data Collection & Preprocessing
* **API Integration:** Automated downloads of the 207,603 bp BRAF reference genomic region from the Ensembl REST API (GRCh38, chr7).
* **Clinical Annotation:** Fetches and processes **919 ClinVar variants** matching the BRAF gene coordinates, complete with clinical significance classifications (pathogenic vs. benign).
* **Splits:** Generates 2,071 training windows (512 bp context size, stride 100) and 866 clinical variant pairs (WT/MUT sequences).

### 2. Model Training
* Pre-training is executed for **50 epochs** on NVIDIA GPUs utilizing CUDA.
* Optimization is performed using **AdamW** with a cosine learning rate decay scheduler.
* The character-level training converges successfully, with cross-entropy loss dropping from **~7.9 (stuck) down to ~1.28** at final epochs.

### 3. Variant Scoring & Evaluation
* Variant pairs are masked, processed through the trained model, and scored using LLR.
* Evaluation metrics include generating **Receiver Operating Characteristic (ROC)** curves, **Precision-Recall (PR)** curves, **Confusion Matrices**, and **Score Distribution** plots to measure separation between ClinVar pathogenic and benign mutations.

---

## 📂 Repository Structure

```
G_game/
├── data/                    # Local clinical and reference downloads (Ensembl, ClinVar)
├── models/                  # PyTorch model architecture and saved checkpoints
│   └── checkpoints/
│       ├── best_model.pt    # Best trained Genome Language Model weights
│       └── training_history.png
├── notebooks/               # Interactive prototyping and visualization
├── results/                 # Evaluation output and performance plots
│   ├── scored_variants.csv  # Final variant scores and LLR predictions
│   └── evaluation_results.png
├── src/                     # Source pipeline components
│   ├── data_collection.py   # Reference genome & variant API downloader
│   ├── dataset.py           # PyTorch genome sequence windows & masking loaders
│   ├── model.py             # Custom GenomeLM transformer architecture
│   ├── score_variants.py    # Log-likelihood Ratio scoring & metric plotter
│   ├── tokenizer.py         # Nucleotide-level tokenizer
│   └── train.py             # Training loop orchestration with Cosine LR
├── .gitignore               # Configured to ignore large weights and venv artifacts
├── requirements.txt         # Project package dependencies
├── run_pipeline.py          # Master orchestration script
└── README.md                # Project documentation
```

---

## ⚡ Setup & Replication Guide

### 📋 Prerequisites
* Python 3.10+
* NVIDIA GPU with CUDA installed (highly recommended for training; scoring can be done on CPU)

### 📥 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/dammmmmmmmmmit/G_game.git
cd G_game

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
```

### 📦 2. Install Dependencies
Make sure you install the packages listed in the requirements file:
```bash
pip install -r requirements.txt
```

### 🎯 3. Run the End-to-End Pipeline
To run data collection, tokenization, model pre-training, variant scoring, and evaluation in one command:
```bash
python run_pipeline.py
```

This will automatically:
1. Fetch BRAF data and ClinVar clinical labels.
2. Initialize and pre-train the Genome Language Model.
3. Calculate LLR anomaly scores for all variants.
4. Save predictions in `results/scored_variants.csv`.
5. Generate performance plots in `results/evaluation_results.png`.

---

## 🎓 Academic Attribution & Citations
This work was developed at **Christ University, Bengaluru** within the **Department of Computer Science**. Special thanks to the academic mentors, peers, and institutional facilities that supported this research.

If you find this research or code useful, please cite:
```bibtex
@article{sharma2026braf,
  title={BRAF Gene Mutation Anomaly Detection Using LLR and Sequence Likelihood Modeling},
  author={Sharma, Aditya},
  journal={Department of Computer Science, Christ University},
  year={2026},
  institution={Christ University, Bengaluru},
  note={Google Docs Paper Draft}
}
```

---
*For any inquiries regarding model capabilities, dataset access, or biological methodology, please contact the author at `aditya.sharma@msam.christuniversity.in`.*
