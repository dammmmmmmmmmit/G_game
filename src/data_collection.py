"""
Data Collection Module for BRAF Mutation Detection Project
This script downloads and prepares genomic data from public sources.
"""

import os
import requests
import json
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import time

# Resolve project root (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create data directories if they don't exist
os.makedirs(os.path.join(PROJECT_ROOT, "data/raw"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "data/processed"), exist_ok=True)


def _path(rel):
    """Resolve a relative path against the project root."""
    return os.path.join(PROJECT_ROOT, rel)


def download_braf_reference_sequence():
    """
    Download BRAF gene reference sequence from Ensembl REST API.
    
    BRAF gene location (GRCh38):
    - Chromosome: 7
    - Start: 140,719,327
    - End: 140,924,929
    - Strand: Reverse (-)
    
    We'll download the sequence plus flanking regions for context.
    """
    
    print("=" * 60)
    print("STEP 1: Downloading BRAF Reference Sequence")
    print("=" * 60)
    
    # Ensembl REST API endpoint
    server = "https://rest.ensembl.org"
    
    # BRAF coordinates on GRCh38 (with 1000bp flanking regions)
    chromosome = "7"
    start = 140719327 - 1000  # Add upstream flank
    end = 140924929 + 1000    # Add downstream flank
    
    # API endpoint for sequence retrieval
    endpoint = f"/sequence/region/human/{chromosome}:{start}..{end}:1"
    
    print(f"Fetching sequence from: chr{chromosome}:{start}-{end}")
    print("This may take a moment...")
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.get(
            server + endpoint,
            headers=headers,
            params={"coord_system_version": "GRCh38"}
        )
        
        if response.status_code == 200:
            data = response.json()
            sequence = data['seq']
            
            # Save the sequence
            output_file = _path("data/raw/braf_reference.fasta")
            with open(output_file, 'w') as f:
                f.write(f">BRAF_chr7_{start}_{end}_GRCh38\n")
                # Write sequence in lines of 60 characters
                for i in range(0, len(sequence), 60):
                    f.write(sequence[i:i+60] + "\n")
            
            print(f"[OK] Successfully downloaded {len(sequence)} base pairs")
            print(f"[OK] Saved to: {output_file}")
            
            # Also save as plain text for easy access
            with open(_path("data/raw/braf_reference.txt"), 'w') as f:
                f.write(sequence)
            
            return sequence
            
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error downloading sequence: {e}")
        return None


def download_clinvar_variants():
    """
    Download BRAF variants from ClinVar.
    
    ClinVar contains clinically annotated variants with pathogenicity
    classifications (pathogenic, benign, uncertain significance, etc.)
    
    We'll use the Ensembl Variant API to get known variants in BRAF region.
    """
    
    print("\n" + "=" * 60)
    print("STEP 2: Downloading ClinVar Variants")
    print("=" * 60)
    
    server = "https://rest.ensembl.org"
    
    # Get variants overlapping BRAF gene
    # Using BRAF Ensembl gene ID
    endpoint = "/overlap/id/ENSG00000157764"
    
    headers = {"Content-Type": "application/json"}
    params = {
        "feature": "variation",
        "variant_set": "ClinVar"
    }
    
    print("Fetching ClinVar variants for BRAF gene...")
    
    try:
        response = requests.get(
            server + endpoint,
            headers=headers,
            params=params
        )
        
        if response.status_code == 200:
            variants = response.json()
            
            # Process variants into a structured format
            variant_list = []
            for var in variants:
                variant_list.append({
                    'id': var.get('id', 'unknown'),
                    'start': var.get('start'),
                    'end': var.get('end'),
                    'strand': var.get('strand'),
                    'alleles': var.get('alleles', []),
                    'consequence_type': var.get('consequence_type', 'unknown'),
                    'clinical_significance': var.get('clinical_significance', ['unknown'])
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(variant_list)
            
            # Save to CSV
            output_file = _path("data/raw/clinvar_braf_variants.csv")
            df.to_csv(output_file, index=False)
            
            print(f"[OK] Downloaded {len(variant_list)} variants")
            print(f"[OK] Saved to: {output_file}")
            
            # Print summary
            if 'clinical_significance' in df.columns:
                print("\nClinical Significance Summary:")
                # Flatten the clinical significance lists
                all_sig = []
                for sig_list in df['clinical_significance']:
                    if isinstance(sig_list, list):
                        all_sig.extend(sig_list)
                    else:
                        all_sig.append(sig_list)
                sig_counts = pd.Series(all_sig).value_counts()
                print(sig_counts.head(10))
            
            return df
            
        else:
            print(f"Error: API returned status code {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error downloading variants: {e}")
        return None


def download_gnomad_variants():
    """
    Download population frequency data from gnomAD.
    
    gnomAD contains variant frequencies from large population studies.
    Common variants (high frequency) are typically benign.
    
    Note: gnomAD doesn't have a simple REST API, so we'll use
    an alternative approach through Ensembl's population data.
    """
    
    print("\n" + "=" * 60)
    print("STEP 3: Downloading Population Variant Frequencies")
    print("=" * 60)
    
    server = "https://rest.ensembl.org"
    
    # Get variants with population frequencies
    endpoint = "/overlap/id/ENSG00000157764"
    
    headers = {"Content-Type": "application/json"}
    params = {
        "feature": "variation"
    }
    
    print("Fetching population variants for BRAF gene...")
    
    try:
        response = requests.get(
            server + endpoint,
            headers=headers,
            params=params
        )
        
        if response.status_code == 200:
            variants = response.json()
            
            # Filter for variants with frequency data
            variant_list = []
            for var in variants:
                # Check if variant has minor allele frequency
                maf = var.get('minor_allele_freq')
                if maf is not None:
                    variant_list.append({
                        'id': var.get('id', 'unknown'),
                        'start': var.get('start'),
                        'end': var.get('end'),
                        'alleles': var.get('alleles', []),
                        'minor_allele': var.get('minor_allele'),
                        'minor_allele_freq': maf,
                        'consequence_type': var.get('consequence_type', 'unknown')
                    })
            
            df = pd.DataFrame(variant_list)
            
            output_file = _path("data/raw/population_variants.csv")
            df.to_csv(output_file, index=False)
            
            print(f"[OK] Downloaded {len(variant_list)} variants with frequency data")
            print(f"[OK] Saved to: {output_file}")
            
            return df
            
        else:
            print(f"Error: API returned status code {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error downloading variants: {e}")
        return None


def create_synthetic_training_data(reference_sequence, num_samples=10000):
    """
    Create synthetic training data from reference sequence.
    
    For training the language model, we need many sequence windows.
    We'll extract overlapping windows from the reference sequence.
    
    This represents "normal" genomic grammar that the model will learn.
    """
    
    print("\n" + "=" * 60)
    print("STEP 4: Creating Training Windows")
    print("=" * 60)
    
    import random
    
    window_size = 512  # Length of each training sequence
    stride = 100       # Step between windows
    
    windows = []
    
    # Extract overlapping windows
    for i in range(0, len(reference_sequence) - window_size, stride):
        window = reference_sequence[i:i + window_size]
        
        # Only keep windows with valid bases (A, T, C, G)
        if all(base in 'ATCG' for base in window.upper()):
            windows.append({
                'sequence': window.upper(),
                'start_position': i,
                'end_position': i + window_size,
                'label': 'wildtype'
            })
    
    print(f"[OK] Created {len(windows)} sequence windows")
    
    # Convert to DataFrame
    df = pd.DataFrame(windows)
    
    # Save training data
    output_file = _path("data/processed/training_windows.csv")
    df.to_csv(output_file, index=False)
    print(f"[OK] Saved to: {output_file}")
    
    return df


def create_variant_sequences(reference_sequence, variants_df):
    """
    Create mutant sequences by introducing variants into reference.
    
    For each variant, we create:
    1. Wild-type window (original reference)
    2. Mutant window (with variant substituted)
    
    These pairs will be used to calculate LLR scores.
    """
    
    print("\n" + "=" * 60)
    print("STEP 5: Creating Variant Sequence Pairs")
    print("=" * 60)
    
    # BRAF region start (for coordinate conversion)
    region_start = 140719327 - 1000
    
    window_size = 512
    half_window = window_size // 2
    
    variant_pairs = []
    
    for idx, row in variants_df.iterrows():
        try:
            # Convert genomic position to local position
            genomic_pos = row['start']
            local_pos = genomic_pos - region_start
            
            # Check if position is within our sequence
            if local_pos < half_window or local_pos >= len(reference_sequence) - half_window:
                continue
            
            # Extract window around variant
            window_start = local_pos - half_window
            window_end = local_pos + half_window
            
            wildtype_seq = reference_sequence[window_start:window_end].upper()
            
            # Skip if sequence contains invalid bases
            if not all(base in 'ATCG' for base in wildtype_seq):
                continue
            
            # Get alleles
            alleles = row.get('alleles', [])
            if isinstance(alleles, str):
                alleles = eval(alleles)  # Convert string representation to list
            
            if len(alleles) < 2:
                continue
            
            ref_allele = alleles[0]
            alt_allele = alleles[1] if len(alleles) > 1 else None
            
            if alt_allele is None or len(alt_allele) != 1:
                continue  # Skip non-SNVs for simplicity
            
            # Create mutant sequence
            variant_pos_in_window = half_window
            mutant_seq = (
                wildtype_seq[:variant_pos_in_window] + 
                alt_allele + 
                wildtype_seq[variant_pos_in_window + 1:]
            )
            
            # Determine label based on clinical significance
            clin_sig = row.get('clinical_significance', ['unknown'])
            if isinstance(clin_sig, str):
                clin_sig = [clin_sig]
            
            # Classify as pathogenic or benign
            if any('pathogenic' in str(s).lower() for s in clin_sig):
                label = 'pathogenic'
            elif any('benign' in str(s).lower() for s in clin_sig):
                label = 'benign'
            else:
                label = 'uncertain'
            
            variant_pairs.append({
                'variant_id': row.get('id', f'var_{idx}'),
                'genomic_position': genomic_pos,
                'ref_allele': ref_allele,
                'alt_allele': alt_allele,
                'wildtype_sequence': wildtype_seq,
                'mutant_sequence': mutant_seq,
                'clinical_significance': str(clin_sig),
                'label': label
            })
            
        except Exception as e:
            continue  # Skip problematic variants
    
    print(f"[OK] Created {len(variant_pairs)} variant pairs")
    
    df = pd.DataFrame(variant_pairs)
    
    # Print label distribution
    if len(df) > 0:
        print("\nLabel Distribution:")
        print(df['label'].value_counts())
    
    output_file = _path("data/processed/variant_pairs.csv")
    df.to_csv(output_file, index=False)
    print(f"[OK] Saved to: {output_file}")
    
    return df


def main():
    """
    Main function to run all data collection steps.
    """
    
    print("\n" + "=" * 60)
    print("BRAF MUTATION DETECTION - DATA COLLECTION")
    print("=" * 60 + "\n")
    
    # Step 1: Download reference sequence
    reference_sequence = download_braf_reference_sequence()
    
    if reference_sequence is None:
        print("\n[!] Failed to download reference. Using fallback method...")
        # Fallback: create a sample sequence for testing
        reference_sequence = create_sample_sequence()
    
    # Brief pause to avoid API rate limits
    time.sleep(1)
    
    # Step 2: Download ClinVar variants
    clinvar_df = download_clinvar_variants()
    
    time.sleep(1)
    
    # Step 3: Download population variants
    population_df = download_gnomad_variants()
    
    # Step 4: Create training windows
    training_df = create_synthetic_training_data(reference_sequence)
    
    # Step 5: Create variant pairs (if we have variant data)
    if clinvar_df is not None and len(clinvar_df) > 0:
        variant_pairs_df = create_variant_sequences(reference_sequence, clinvar_df)
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/raw/braf_reference.fasta")
    print("  - data/raw/braf_reference.txt")
    print("  - data/raw/clinvar_braf_variants.csv")
    print("  - data/raw/population_variants.csv")
    print("  - data/processed/training_windows.csv")
    print("  - data/processed/variant_pairs.csv")


def create_sample_sequence():
    """
    Create a sample BRAF-like sequence for testing if API fails.
    This is a fallback only - real data is preferred.
    """
    
    import random
    random.seed(42)
    
    # Create a realistic-looking sequence
    bases = ['A', 'T', 'C', 'G']
    
    # GC content around 40% (typical for human genome)
    weights = [0.3, 0.3, 0.2, 0.2]
    
    sequence = ''.join(random.choices(bases, weights=weights, k=50000))
    
    # Save it
    with open(_path("data/raw/braf_reference.txt"), 'w') as f:
        f.write(sequence)
    
    print(f"[OK] Created sample sequence of {len(sequence)} bp")
    
    return sequence


if __name__ == "__main__":
    main()
