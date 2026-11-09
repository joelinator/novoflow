#!/usr/bin/env python3
"""
NovoFlow Quickstart Example
Demonstrates how to load spectra, configure the beam generator with Knapsack reachability,
and export identified peptides with calibrated FDR estimation.
"""

import sys
from pathlib import Path

# Add repo root to path if running directly from examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from novoflow import NovoFlowPipeline
from novoflow.io import read_mgf, export_identifications


def main():
    example_mgf = Path(__file__).parent / "sample.mgf"
    output_tsv = Path(__file__).parent / "sample_results.tsv"

    print(f"Loading spectra from {example_mgf}...")
    spectra = list(read_mgf(str(example_mgf)))
    print(f"Loaded {len(spectra)} spectra.")

    # Initialize NovoFlow pipeline
    # In production, pass checkpoint_path="/path/to/novoflow_weights.pt"
    # When checkpoint_path is None, a randomly initialized model is instantiated for testing/dry-run.
    print("\nInitializing NovoFlow Pipeline...")
    pipeline = NovoFlowPipeline(
        checkpoint_path=None,
        device="cpu",
        top_k_lengths=3,
        samples_per_length=2,
        flow_steps=16,
    )

    print("\nRunning de novo sequencing...")
    results = pipeline.predict_batch(spectra)

    print("\nResults:")
    for r in results:
        print(f"  [{r['title']}] Seq: {r['sequence']} | Score: {r['novoflow_score']:.3f} | PEP: {r['pep']:.4f} | q-value: {r['q_value']:.4f}")

    print(f"\nExporting to TSV ({output_tsv})...")
    export_identifications(results, str(output_tsv), format="tsv")
    print("Done! Export complete.")


if __name__ == "__main__":
    main()
