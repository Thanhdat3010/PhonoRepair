"""Word Error Rate (WER) and statistical bootstrap significance testing."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    num_reference_words: int

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.num_reference_words == 0:
            return 0.0 if self.total_errors == 0 else 1.0
        return self.total_errors / self.num_reference_words


def compute_wer_counts(reference_tokens: List[str], hypothesis_tokens: List[str]) -> EditCounts:
    """Compute exact Substitution, Deletion, and Insertion counts via Wagner-Fischer DP."""
    R = len(reference_tokens)
    H = len(hypothesis_tokens)

    # DP table storing (cost, S, D, I)
    dp = [[(0, 0, 0, 0)] * (H + 1) for _ in range(R + 1)]

    for i in range(1, R + 1):
        dp[i][0] = (i, 0, i, 0)  # Deletions

    for j in range(1, H + 1):
        dp[0][j] = (j, 0, 0, j)  # Insertions

    for i in range(1, R + 1):
        for j in range(1, H + 1):
            ref_w = reference_tokens[i - 1]
            hyp_w = hypothesis_tokens[j - 1]

            if ref_w == hyp_w:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                cost_sub, s_sub, d_sub, i_sub = dp[i - 1][j - 1]
                sub_opt = (cost_sub + 1, s_sub + 1, d_sub, i_sub)

                cost_del, s_del, d_del, i_del = dp[i - 1][j]
                del_opt = (cost_del + 1, s_del, d_del + 1, i_del)

                cost_ins, s_ins, d_ins, i_ins = dp[i][j - 1]
                ins_opt = (cost_ins + 1, s_ins, d_ins, i_ins + 1)

                dp[i][j] = min(sub_opt, del_opt, ins_opt, key=lambda x: x[0])

    _, s, d, ins = dp[R][H]
    return EditCounts(substitutions=s, deletions=d, insertions=ins, num_reference_words=R)


def compute_corpus_wer(counts_list: List[EditCounts]) -> float:
    """Compute corpus-level WER = sum(S + D + I) / sum(N)."""
    total_errors = sum(c.total_errors for c in counts_list)
    total_ref = sum(c.num_reference_words for c in counts_list)
    if total_ref == 0:
        return 0.0
    return total_errors / total_ref


def paired_cluster_bootstrap(
    speaker_results: Dict[str, List[Tuple[EditCounts, EditCounts]]],
    num_replicates: int = 1000,
    seed: int = 20261013,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Speaker-clustered paired bootstrap test for delta WER = WER_repair - WER_base.

    Args:
        speaker_results: Dict mapping speaker_id -> list of (base_counts, repair_counts)
        num_replicates: Number of bootstrap cluster resamples
        seed: Random seed
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Dictionary with delta_wer_mean, ci_lower, ci_upper, p_value, is_significant
    """
    rng = np.random.default_rng(seed)
    speakers = list(speaker_results.keys())
    n_speakers = len(speakers)
    assert n_speakers > 0, "No speakers provided"

    # Precompute per-speaker totals for fast sampling
    spk_base_err = []
    spk_rep_err = []
    spk_ref_words = []

    for spk in speakers:
        pairs = speaker_results[spk]
        b_err = sum(p[0].total_errors for p in pairs)
        r_err = sum(p[1].total_errors for p in pairs)
        ref_n = sum(p[0].num_reference_words for p in pairs)

        spk_base_err.append(b_err)
        spk_rep_err.append(r_err)
        spk_ref_words.append(ref_n)

    spk_base_err = np.array(spk_base_err, dtype=np.float64)
    spk_rep_err = np.array(spk_rep_err, dtype=np.float64)
    spk_ref_words = np.array(spk_ref_words, dtype=np.float64)

    delta_wer_replicates = []

    for _ in range(num_replicates):
        sampled_indices = rng.choice(n_speakers, size=n_speakers, replace=True)
        tot_ref = np.sum(spk_ref_words[sampled_indices])
        if tot_ref == 0:
            continue

        wer_base = np.sum(spk_base_err[sampled_indices]) / tot_ref
        wer_rep = np.sum(spk_rep_err[sampled_indices]) / tot_ref
        delta_wer_replicates.append(wer_rep - wer_base)

    delta_array = np.array(delta_wer_replicates)
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(delta_array, lower_pct))
    ci_upper = float(np.percentile(delta_array, upper_pct))
    point_delta = float(np.mean(delta_array))

    # Empirical p-value for H0: delta >= 0 (repair does not improve)
    p_value = float(np.mean(delta_array >= 0.0))

    return {
        "point_delta_wer": point_delta,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "p_value": p_value,
        "is_significant_improvement": ci_upper < 0.0,
    }
