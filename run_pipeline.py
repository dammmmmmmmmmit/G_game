"""
Main Pipeline Script — runs the complete BRAF mutation detection pipeline.
"""
import os, sys, subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PROJECT_ROOT, "src")

def run_step(name, script):
    print("\n" + "=" * 70 + f"\nSTEP: {name}\n" + "=" * 70)
    rc = subprocess.run([sys.executable, script], cwd=SRC, capture_output=False).returncode
    print(f"{'[OK]' if rc == 0 else '[!]'} {name} {'completed' if rc == 0 else 'had issues'}")
    return rc

def main():
    print("=" * 70 + "\nBRAF MUTATION ANOMALY DETECTION PIPELINE\n" + "=" * 70)
    print(f"Start: {datetime.now()}")
    steps = [
        ("Data Collection", os.path.join(SRC, "data_collection.py")),
        ("Model Training",  os.path.join(SRC, "train.py")),
        ("Variant Scoring", os.path.join(SRC, "score_variants.py")),
    ]
    results = {}
    for name, script in steps:
        results[name] = run_step(name, script) if os.path.exists(script) else -1
    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    for n, rc in results.items():
        print(f"  {n}: {'[OK]' if rc == 0 else '[!]'}")
    print(f"End: {datetime.now()}")

if __name__ == "__main__":
    main()
