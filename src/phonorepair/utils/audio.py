"""Audio input/output utilities for PhonoRepair."""

import io
from typing import Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_and_resample_audio(
    audio_source: Union[str, bytes, dict], target_sample_rate: int = 16000
) -> Tuple[torch.Tensor, int]:
    """Load an audio file or bytes and resample to target_sample_rate (default 16kHz).

    Supports:
    - Filepath (str)
    - Raw audio bytes
    - HuggingFace datasets audio dict ({"array": np.ndarray, "sampling_rate": int})
    """
    if isinstance(audio_source, dict) and "array" in audio_source:
        array = audio_source["array"]
        sr = audio_source["sampling_rate"]
        tensor = torch.from_numpy(array).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
    elif isinstance(audio_source, bytes):
        data, sr = sf.read(io.BytesIO(audio_source), dtype="float32")
        tensor = torch.from_numpy(data).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.transpose(0, 1)
    elif isinstance(audio_source, str):
        tensor, sr = torchaudio.load(audio_source)
    else:
        raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

    # Convert to mono if multichannel
    if tensor.shape[0] > 1:
        tensor = torch.mean(tensor, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        tensor = resampler(tensor)

    # Squeeze to 1D: (num_samples,)
    return tensor.squeeze(0), target_sample_rate
