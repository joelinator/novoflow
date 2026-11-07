"""
NovoFlow Command Line Interface (CLI).
Provides user-friendly commands: predict, calibrate, info.
"""

from __future__ import annotations
import argparse
import sys
import time
import torch

from novoflow import __version__
from novoflow.constants import VOCAB, CANONICAL_AAS, AA_MASSES
from novoflow.pipeline import NovoFlowPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="novoflow",
        description="NovoFlow: High-Throughput De Novo Peptide Sequencing via Discrete Flow Matching",
    )
    parser.add_argument("--version", action="version", version=f"novoflow {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="NovoFlow subcommands")

    # Command: predict
    pred_parser = subparsers.add_parser("predict", help="Sequence an MS/MS run (.mgf or .mzML)")
    pred_parser.add_argument("-i", "--input", required=True, help="Input tandem MS file (.mgf or .mzML)")
    pred_parser.add_argument("-o", "--output", required=True, help="Output file (.tsv, .csv, or .mzTab)")
    pred_parser.add_argument("-c", "--checkpoint", default=None, help="Trained model checkpoint (.ckpt)")
    pred_parser.add_argument("--fdr", type=float, default=0.01, help="FDR threshold for report (default: 0.01)")
    pred_parser.add_argument("--filter-fdr", action="store_true", help="Filter output to only FDR-passing hits")
    pred_parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference (default: 128)")
    pred_parser.add_argument("--device", default=None, help="Device to run inference on ('cuda' or 'cpu')")
    pred_parser.add_argument("--top-k", type=int, default=5, help="Top-K predicted lengths to explore (default: 5)")
    pred_parser.add_argument("--samples", type=int, default=2, help="Samples per length beam (default: 2)")

    # Command: info
    info_parser = subparsers.add_parser("info", help="Display NovoFlow configuration and system environment")

    return parser


def cmd_info():
    print(f"============================================================")
    print(f"NovoFlow v{__version__}")
    print(f"Continuous-Time Discrete Flow Matching for De Novo Sequencing")
    print(f"============================================================")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name       : {torch.cuda.get_device_name(0)}")
        print(f"Device Memory     : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"Vocabulary Size   : {len(VOCAB)} tokens")
    print(f"Canonical Residues: {', '.join(CANONICAL_AAS)}")
    print(f"============================================================")


def cmd_predict(args):
    t0 = time.time()
    print(f"[*] Initializing NovoFlow Pipeline (device={args.device or ('cuda' if torch.cuda.is_available() else 'cpu')})...")
    pipeline = NovoFlowPipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
        top_k_lengths=args.top_k,
        samples_per_length=args.samples,
    )

    print(f"[*] Reading and processing '{args.input}'...")
    results = pipeline.predict_file(
        input_path=args.input,
        output_path=args.output,
        fdr_threshold=args.fdr,
        batch_size=args.batch_size,
        filter_fdr=args.filter_fdr,
    )

    elapsed = time.time() - t0
    n_total = len(results)
    rate = n_total / max(1e-4, elapsed)
    n_passed = sum(1 for r in results if r.get("passes_1pct_fdr", False))

    print(f"\n[+] Sequencing Complete!")
    print(f"    Total Spectra Processed: {n_total:,}")
    print(f"    Elapsed Time           : {elapsed:.2f} s ({rate:.1f} spectra/s)")
    print(f"    Hits Passing 1% FDR    : {n_passed:,} ({(n_passed / max(1, n_total))*100:.1f}%)")
    print(f"    Results exported to    : {args.output}\n")


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if args.command == "info":
        cmd_info()
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
