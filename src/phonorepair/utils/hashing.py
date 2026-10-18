"""Deterministic hashing and data partitioning for PhonoRepair.

Implements the protocol from Proposal v7.1 Section 10.1:
- SHA256(seed || corpus || speaker_id) determines the deterministic sort order.
- train.clean.100: first 20% speakers for basis-fit, remaining 80% for dict-fit.
- train.clean.360: 100 speakers selection, 100 calibration, 100 stress-validation, rest holdout.
"""

import hashlib
from typing import Dict, List, Tuple


def compute_speaker_hash(seed: int, corpus: str, speaker_id: str) -> str:
    """Compute deterministic SHA256 hash for a given speaker ID."""
    token = f"{seed}|{corpus}|{speaker_id}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def sort_speakers_by_hash(seed: int, corpus: str, speaker_ids: List[str]) -> List[str]:
    """Sort unique speaker IDs deterministically using the SHA256 hash."""
    unique_speakers = sorted(list(set(speaker_ids)))
    return sorted(unique_speakers, key=lambda spk: compute_speaker_hash(seed, corpus, spk))


def split_train_clean_100(
    speaker_ids: List[str], seed: int = 17, corpus: str = "librispeech"
) -> Tuple[List[str], List[str]]:
    """Partition train.clean.100 speakers into:
    - basis_fit: first 20%
    - dict_fit: remaining 80%
    """
    sorted_spks = sort_speakers_by_hash(seed, corpus, speaker_ids)
    n_total = len(sorted_spks)
    n_basis = max(1, int(n_total * 0.20))
    basis_fit = sorted_spks[:n_basis]
    dict_fit = sorted_spks[n_basis:]
    return basis_fit, dict_fit


def split_train_clean_360(
    speaker_ids: List[str],
    seed: int = 17,
    corpus: str = "librispeech",
    n_selection: int = 100,
    n_calibration: int = 100,
    n_stress: int = 100,
) -> Dict[str, List[str]]:
    """Partition train.clean.360 speakers into:
    - clean_selection: n_selection
    - calibration: n_calibration
    - clean_stress_validation: n_stress
    - clean_speaker_holdout: remaining
    """
    sorted_spks = sort_speakers_by_hash(seed, corpus, speaker_ids)
    idx1 = min(len(sorted_spks), n_selection)
    idx2 = min(len(sorted_spks), idx1 + n_calibration)
    idx3 = min(len(sorted_spks), idx2 + n_stress)

    return {
        "clean_selection": sorted_spks[:idx1],
        "calibration": sorted_spks[idx1:idx2],
        "clean_stress_validation": sorted_spks[idx2:idx3],
        "clean_speaker_holdout": sorted_spks[idx3:],
    }


def assign_split(speaker_id: str, split_dict: Dict[str, List[str]]) -> str:
    """Return the partition name for a given speaker ID."""
    for split_name, spk_list in split_dict.items():
        if speaker_id in spk_list:
            return split_name
    return "unassigned"
