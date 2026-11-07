"""
NovoFlow High-Level Inference Pipeline.
Provides turn-key de novo sequencing from .mgf or .mzML to FDR-filtered identifications.
"""

from __future__ import annotations
import os
import time
from typing import Any
import torch

from novoflow.model.transformer import SpectrumEncoder, DFMPeptideDecoder, PeptideLengthClassifier
from novoflow.model.hub import get_default_checkpoint
from novoflow.reachability.knapsack import ExactReachabilityDP
from novoflow.inference.beam_generator import BeamGenerator
from novoflow.reranker.listnet import ListNetReranker
from novoflow.reranker.features import extract_features
from novoflow.fdr.engine import FDREngine
from novoflow.io.mgf import read_mgf_batches, MGFSpectrum
from novoflow.io.mzml import read_mzml
from novoflow.io.export import export_tsv, export_csv, export_mztab


class NovoFlowPipeline:
    """
    End-to-end de novo peptide sequencing pipeline.
    """
    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
        flow_steps: int = 20,
        top_k_lengths: int = 5,
        samples_per_length: int = 2,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.flow_steps = flow_steps
        self.top_k_lengths = top_k_lengths
        self.samples_per_length = samples_per_length

        # 1. Initialize core neural modules
        self.encoder = SpectrumEncoder(model_dim=512, num_layers=6).to(self.device)
        self.decoder = DFMPeptideDecoder(spec_dim=512, emb_dim=512, n_decoder_blocks=6).to(self.device)
        self.length_predictor = PeptideLengthClassifier(in_dim=512, hidden_dim=256).to(self.device)

        # 2. Initialize DP knapsack reachability
        self.reachability_dp = ExactReachabilityDP.get_default(device=self.device)

        # 3. Initialize beam generator & reranker
        self.beam_generator = BeamGenerator(
            spectrum_encoder=self.encoder,
            decoder=self.decoder,
            length_predictor=self.length_predictor,
            reachability_dp=self.reachability_dp,
            flow_steps=self.flow_steps,
            device=self.device,
        )
        self.reranker = ListNetReranker()
        self.fdr_engine = FDREngine()

        # 4. Load weights (defaulting to official MassIVE-KB pretrained checkpoint)
        if checkpoint_path is None:
            try:
                checkpoint_path = get_default_checkpoint()
            except Exception as e:
                print(f"[*] Warning: Could not resolve default checkpoint ({e}). Running without pretrained weights.")
                checkpoint_path = None

        if checkpoint_path is not None and str(checkpoint_path).lower() != "none" and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load trained weights from a PyTorch Lightning or custom checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        def clean_sd(prefix: str):
            res = {}
            for k, v in sd.items():
                if k.startswith(prefix + "."):
                    res[k[len(prefix) + 1:]] = v
            return res

        enc_sd = clean_sd("spectrum_encoder")
        dec_sd = clean_sd("decoder")
        lp_sd = clean_sd("length_predictor")

        if enc_sd:
            self.encoder.load_state_dict(enc_sd, strict=False)
        if dec_sd:
            self.decoder.load_state_dict(dec_sd, strict=False)
        if lp_sd:
            self.length_predictor.load_state_dict(lp_sd, strict=False)

    def predict_spectra(
        self,
        spectra: list[MGFSpectrum],
        fdr_threshold: float = 0.01,
    ) -> list[dict[str, Any]]:
        """Run de novo sequencing on an in-memory list of MGFSpectrum objects."""
        if not spectra:
            return []

        # Generate candidates: (B, K*S)
        cand_lists = self.beam_generator.generate_candidates(
            spectra=spectra,
            top_k_lengths=self.top_k_lengths,
            samples_per_length=self.samples_per_length,
        )

        top_identifications = []
        for spec, cands in zip(spectra, cand_lists):
            # Extract features for reranking
            for c in cands:
                c["features"] = extract_features(
                    candidate_seq=c["sequence"],
                    precursor_mass=spec.precursor_neutral_mass,
                    expected_length=c["length"],
                    length_log_prob=c.get("length_log_prob", 0.0),
                    mean_log_prob=c.get("mean_log_prob", 0.0),
                    frag_metrics=c.get("frag_metrics"),
                )

            # Rerank candidates with ListNet
            reranked = self.reranker.rerank_candidates(cands)
            top1 = reranked[0] if reranked else {}

            top_identifications.append({
                "raw_file": spec.raw_file,
                "scan_num": spec.scan_num,
                "title": spec.title,
                "rt": spec.rt,
                "precursor_mz": spec.precursor_mz,
                "charge": spec.charge,
                "precursor_neutral_mass": spec.precursor_neutral_mass,
                "sequence": top1.get("sequence", ""),
                "calc_mass": top1.get("calc_mass", 0.0),
                "delta_m": top1.get("delta_m", 0.0),
                "ppm_error": top1.get("ppm_error", 0.0),
                "novoflow_score": top1.get("novoflow_score", 0.0),
                "frag_score": top1.get("frag_score", 0.0),
                "theo_coverage": top1.get("theo_coverage", 0.0),
                "ladder_score": top1.get("ladder_score", 0.0),
                "explained_intensity": top1.get("explained_intensity", 0.0),
                "b_matched": top1.get("b_matched", 0),
                "y_matched": top1.get("y_matched", 0),
            })

        # Annotate with PEP and q-values
        scores = [r["novoflow_score"] for r in top_identifications]
        peps, q_vals = self.fdr_engine.compute_q_values_empirical(scores)
        for r, pep, q in zip(top_identifications, peps, q_vals):
            r["pep"] = pep
            r["q_value"] = q
            r["passes_1pct_fdr"] = (q <= 0.01)

        return top_identifications

    predict_batch = predict_spectra

    def predict_file(
        self,
        input_path: str,
        output_path: str | None = None,
        fdr_threshold: float = 0.01,
        batch_size: int = 128,
        filter_fdr: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Sequences an entire .mgf or .mzML run with streaming batches and exports results.
        """
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".mgf":
            batch_iter = read_mgf_batches(input_path, batch_size=batch_size)
        elif ext in (".mzml", ".xml"):
            # Stream mzML in batches
            def _mzml_batches():
                b = []
                for s in read_mzml(input_path):
                    b.append(s)
                    if len(b) >= batch_size:
                        yield b
                        b = []
                if b:
                    yield b
            batch_iter = _mzml_batches()
        else:
            raise ValueError(f"Unsupported file format: {ext} (must be .mgf or .mzml)")

        all_identifications = []
        for batch in batch_iter:
            ids = self.predict_spectra(batch, fdr_threshold=fdr_threshold)
            all_identifications.extend(ids)

        # Recalculate global FDR over all batches in run
        if all_identifications:
            scores = [r["novoflow_score"] for r in all_identifications]
            peps, q_vals = self.fdr_engine.compute_q_values_empirical(scores)
            for r, pep, q in zip(all_identifications, peps, q_vals):
                r["pep"] = pep
                r["q_value"] = q
                r["passes_1pct_fdr"] = (q <= 0.01)

        results_to_save = all_identifications
        if filter_fdr:
            results_to_save = [r for r in all_identifications if r.get("q_value", 1.0) <= fdr_threshold]

        if output_path:
            out_ext = os.path.splitext(output_path)[1].lower()
            if out_ext == ".tsv":
                export_tsv(output_path, results_to_save)
            elif out_ext == ".csv":
                export_csv(output_path, results_to_save)
            elif out_ext == ".mztab":
                export_mztab(output_path, results_to_save)
            else:
                export_tsv(output_path + ".tsv", results_to_save)

        return results_to_save
