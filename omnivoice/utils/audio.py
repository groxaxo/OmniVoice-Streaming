#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audio I/O and processing utilities.

Provides functions for loading, resampling, silence removal, chunking,
cross-fading, and format conversion. Used by ``OmniVoice.generate()`` during
inference post-processing.
"""

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, detect_nonsilent, split_on_silence


def load_audio(audio_path: str, sampling_rate: int):
    """
    Load the waveform with torchaudio and resampling if needed.

    Parameters:
        audio_path: path of the audio.
        sampling_rate: target sampling rate.

    Returns:
        Loaded prompt waveform with target sampling rate,
        PyTorch tensor of shape (1, T)
    """
    try:
        waveform, prompt_sampling_rate = torchaudio.load(
            audio_path, backend="soundfile"
        )
    except (RuntimeError, OSError):
        # Fallback via pydub+ffmpeg for formats torchaudio can't handle
        aseg = AudioSegment.from_file(audio_path)
        audio_data = np.array(aseg.get_array_of_samples()).astype(np.float32) / 32768.0
        if aseg.channels == 1:
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_data.reshape(-1, aseg.channels).T)
        prompt_sampling_rate = aseg.frame_rate

    if prompt_sampling_rate != sampling_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=prompt_sampling_rate,
            new_freq=sampling_rate,
        )
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform


def remove_silence(
    audio: torch.Tensor,
    sampling_rate: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
):
    """
    Remove middle silences longer than mid_sil ms, and edge silences longer than edge_sil ms

    Parameters:
        audio: PyTorch tensor with shape (C, T).
        sampling_rate: sampling rate of the audio.
        mid_sil: the duration of silences in the middle of audio to be removed in ms.
            if mid_sil <= 0, no middle silence will be removed.
        edge_sil: the duration of silences in the edge of audio to be removed in ms.
        trail_sil: the duration of added trailing silence in ms.

    Returns:
        PyTorch tensor with shape (C, T), where C is number of channels
            and T is number of audio samples
    """
    # Load audio file
    wave = tensor_to_audiosegment(audio, sampling_rate)

    if mid_sil > 0:
        # Split audio using silences longer than mid_sil
        non_silent_segs = split_on_silence(
            wave,
            min_silence_len=mid_sil,
            silence_thresh=-50,
            keep_silence=mid_sil,
            seek_step=10,
        )

        # Concatenate all non-silent segments
        wave = AudioSegment.silent(duration=0)
        for seg in non_silent_segs:
            wave += seg

    # Remove silence longer than 0.1 seconds in the begining and ending of wave
    wave = remove_silence_edges(wave, lead_sil, trail_sil, -50)

    # Convert to PyTorch tensor
    return audiosegment_to_tensor(wave)


def remove_silence_edges(
    audio: AudioSegment,
    lead_sil: int = 100,
    trail_sil: int = 300,
    silence_threshold: float = -50,
):
    """
    Remove edge silences longer than `keep_silence` ms.

    Parameters:
        audio: an AudioSegment object.
        keep_silence: kept silence in the edge.
        only_edge: If true, only remove edge silences.
        silence_threshold: the threshold of silence.

    Returns:
        An AudioSegment object
    """
    # Remove heading silence
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - lead_sil)
    audio = audio[start_idx:]

    # Remove trailing silence
    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - trail_sil)
    audio = audio[start_idx:]
    audio = audio.reverse()

    return audio


def trim_trailing_artifact(
    audio: torch.Tensor,
    sampling_rate: int,
    silence_thresh: float = -50,
    min_silence_ms: int = 15,
    max_artifact_ms: int = 200,
) -> torch.Tensor:
    """
    Trim a short energy burst that appears after a silence gap at the end of audio.

    Masked diffusion models occasionally over-generate a few extra frames beyond
    the last real phoneme, producing a brief (~80ms) artifact burst that is above
    the silence threshold and therefore not removed by remove_silence_edges().

    The detection pattern is:
        [real speech] → [silence ≥ min_silence_ms] → [short burst ≤ max_artifact_ms]
                                                           ↑ trim from here

    Parameters:
        audio: PyTorch tensor of shape (C, T)
        sampling_rate: sample rate of the audio
        silence_thresh: dBFS level below which a frame is "silent" (default -50)
        min_silence_ms: minimum gap length in ms to consider it a silence boundary (default 15)
        max_artifact_ms: maximum length in ms of the post-silence burst to trim

    Returns:
        PyTorch tensor (C, T') with trailing artifact removed (if detected)
    """
    aseg = tensor_to_audiosegment(audio, sampling_rate)
    duration_ms = len(aseg)
    if duration_ms < min_silence_ms + max_artifact_ms:
        return audio

    chunk_ms = 10  # analyse in 10ms chunks
    # Scan the last (max_artifact_ms + 200ms) to keep it cheap
    scan_start = max(0, duration_ms - max_artifact_ms - 200)
    window = aseg[scan_start:]

    levels = []
    for i in range(0, len(window), chunk_ms):
        chunk = window[i : i + chunk_ms]
        levels.append((scan_start + i, chunk.dBFS))

    # Walk backwards: find [silence gap] → [short burst] at end
    trim_at = None
    in_burst = False
    burst_start = None
    burst_len = 0
    silence_len = 0

    for pos, lvl in reversed(levels):
        if not in_burst:
            if lvl > silence_thresh:
                in_burst = True
                burst_start = pos
                burst_len = chunk_ms
                silence_len = 0
        else:
            if lvl > silence_thresh:
                burst_len += chunk_ms
                if burst_len > max_artifact_ms:
                    # Too long to be an artifact — it's real speech
                    break
            else:
                silence_len += chunk_ms
                if silence_len >= min_silence_ms:
                    # Found: silence gap followed by a short burst → artifact
                    trim_at = burst_start
                    break

    if trim_at is not None and trim_at > 0:
        trim_samples = int(trim_at * sampling_rate / 1000)
        audio = audio[:, :trim_samples]

    return audio


def audiosegment_to_tensor(aseg):
    """
    Convert a pydub.AudioSegment to PyTorch audio tensor
    """
    audio_data = np.array(aseg.get_array_of_samples())

    # Convert to float32 and normalize to [-1, 1] range
    audio_data = audio_data.astype(np.float32) / 32768.0

    # Handle channels
    if aseg.channels == 1:
        # Mono channel: add channel dimension (T) -> (1, T)
        tensor_data = torch.from_numpy(audio_data).unsqueeze(0)
    else:
        # Multi-channel: reshape to (C, T)
        tensor_data = torch.from_numpy(audio_data.reshape(-1, aseg.channels).T)

    return tensor_data


def tensor_to_audiosegment(tensor, sample_rate):
    """
    Convert a PyTorch audio tensor to pydub.AudioSegment

    Parameters:
        tensor: Tensor with shape (C, T), where C is the number of channels
            and T is the time steps
        sample_rate: Audio sample rate
    """
    # Convert tensor to numpy array
    assert isinstance(tensor, torch.Tensor)
    audio_np = tensor.cpu().numpy()

    # Convert to int16 type (common format for pydub)
    # Assumes tensor values are in [-1, 1] range as floating point
    audio_np = (audio_np * 32768.0).clip(-32768, 32767).astype(np.int16)

    # Convert to byte stream
    # For multi-channel audio, pydub requires interleaved format
    # (e.g., left-right-left-right)
    if audio_np.shape[0] > 1:
        # Convert to interleaved format
        audio_np = audio_np.transpose(1, 0).flatten()
    audio_bytes = audio_np.tobytes()

    # Create AudioSegment
    audio_segment = AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=tensor.shape[0],
    )

    return audio_segment


def fade_and_pad_audio(
    audio: torch.Tensor,
    pad_duration: float = 0.1,
    fade_duration: float = 0.1,
    sample_rate: int = 24000,
) -> torch.Tensor:
    """
    Applies a smooth fade-in and fade-out to the audio, and then pads both sides
    with pure silence to prevent abrupt starts and ends (clicks/pops).

    Args:
        audio: PyTorch tensor of shape (C, T) containing audio data.
        pad_duration: Duration of pure silence to add to each end (in seconds).
        fade_duration: Duration of the fade-in/out curve (in seconds).
        sample_rate: Audio sampling rate.

    Returns:
        Processed sequence tensor with shape (C, T_new)
    """
    if audio.shape[-1] == 0:
        return audio

    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)

    processed = audio.clone()

    if fade_samples > 0:
        k = min(fade_samples, processed.shape[-1] // 2)

        if k > 0:
            fade_in = torch.linspace(
                0, 1, k, device=processed.device, dtype=processed.dtype
            )[None, :]
            processed[..., :k] = processed[..., :k] * fade_in

            fade_out = torch.linspace(
                1, 0, k, device=processed.device, dtype=processed.dtype
            )[None, :]
            processed[..., -k:] = processed[..., -k:] * fade_out

    if pad_samples > 0:
        silence = torch.zeros(
            (processed.shape[0], pad_samples),
            dtype=processed.dtype,
            device=processed.device,
        )
        processed = torch.cat([silence, processed, silence], dim=-1)

    return processed


def trim_long_audio(
    audio: torch.Tensor,
    sampling_rate: int,
    max_duration: float = 15.0,
    min_duration: float = 3.0,
    trim_threshold: float = 20.0,
) -> torch.Tensor:
    """Trim audio to <= max_duration by splitting at the largest silence gap.

    Only trims when the audio exceeds *trim_threshold* seconds.

    Args:
        audio: Audio tensor of shape (C, T).
        sampling_rate: Audio sampling rate.
        max_duration: Maximum duration in seconds.
        min_duration: Minimum duration in seconds.
        trim_threshold: Only trim if audio is longer than this (seconds).

    Returns:
        Trimmed audio tensor.
    """
    duration = audio.size(-1) / sampling_rate
    if duration <= trim_threshold:
        return audio

    seg = tensor_to_audiosegment(audio, sampling_rate)
    nonsilent = detect_nonsilent(
        seg, min_silence_len=100, silence_thresh=-40, seek_step=10
    )
    if not nonsilent:
        return audio

    max_ms = int(max_duration * 1000)
    min_ms = int(min_duration * 1000)

    # Walk through speech regions; at each gap pick the latest split <= max_duration
    best_split = 0
    for start, end in nonsilent:
        if start > best_split and start <= max_ms:
            best_split = start
        if end > max_ms:
            break

    if best_split < min_ms:
        best_split = min(max_ms, len(seg))

    trimmed = seg[:best_split]
    return audiosegment_to_tensor(trimmed)


def cross_fade_chunks(
    chunks: list[torch.Tensor],
    sample_rate: int,
    silence_duration: float = 0.3,
) -> torch.Tensor:
    """Concatenate audio chunks with a short silence gap and fade at boundaries.

    Each boundary is structured as: fade-out tail → silence buffer → fade-in head.
    This avoids click artifacts from direct concatenation or overlapping mismatch.

    Args:
        chunks: List of audio tensors, each (C, T).
        sample_rate: Audio sample rate.
        silence_duration: Total silence gap duration in seconds.

    Returns:
        Merged audio tensor (C, T_total).
    """
    if len(chunks) == 1:
        return chunks[0]

    total_n = int(silence_duration * sample_rate)
    fade_n = total_n // 3
    silence_n = fade_n  # middle silent gap

    # Single-pass: collect all segments in a flat list and cat once at the end.
    # The iterative approach grew `merged` by torch.cat on every iteration,
    # which copies all previously accumulated audio O(N) times — O(N²) total.
    # Here we process each chunk independently and do one final torch.cat.
    parts: list[torch.Tensor] = []
    for idx, chunk in enumerate(chunks):
        dev, dt = chunk.device, chunk.dtype
        piece = chunk.clone()

        # Fade-out on the tail of every chunk except the last
        if idx < len(chunks) - 1:
            fout_n = min(fade_n, piece.size(-1))
            if fout_n > 0:
                w_out = torch.linspace(1, 0, fout_n, device=dev, dtype=dt)[None, :]
                piece[..., -fout_n:] = piece[..., -fout_n:] * w_out

        # Fade-in on the head of every chunk except the first
        if idx > 0:
            fin_n = min(fade_n, piece.size(-1))
            if fin_n > 0:
                w_in = torch.linspace(0, 1, fin_n, device=dev, dtype=dt)[None, :]
                piece[..., :fin_n] = piece[..., :fin_n] * w_in

        parts.append(piece)

        # Silence gap after every chunk except the last
        if idx < len(chunks) - 1:
            parts.append(torch.zeros(chunk.shape[0], silence_n, device=dev, dtype=dt))

    return torch.cat(parts, dim=-1)
