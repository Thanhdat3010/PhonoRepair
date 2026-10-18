"""Evaluation, metrics, normalization, and reporting for PhonoRepair.

Implements Proposal Section 11:
- normalizer: Normalizer v1 for fair symmetric text comparison
- metrics: Word Error Rate (WER) with S, D, I counts, paired difference, bootstrap CI
- reject_tracker: Table D diagnostic accounting (first-exit reasons, harms vs corrections)
- table_builder: LaTeX and Markdown generator for Table 1 and Table 2
"""

from .normalizer import normalize_text_v1
from .metrics import compute_wer_counts, compute_corpus_wer, paired_cluster_bootstrap
from .reject_tracker import RejectTracker
from .table_builder import export_table_1_markdown, export_table_1_latex
from .runner import EvaluationRunner

from .perturbation import add_acoustic_noise, perturb_latent_frame, perturb_latent_sequence

__all__ = [
    "normalize_text_v1",
    "compute_wer_counts",
    "compute_corpus_wer",
    "paired_cluster_bootstrap",
    "RejectTracker",
    "export_table_1_markdown",
    "export_table_1_latex",
    "EvaluationRunner",
    "add_acoustic_noise",
    "perturb_latent_frame",
    "perturb_latent_sequence",
]
