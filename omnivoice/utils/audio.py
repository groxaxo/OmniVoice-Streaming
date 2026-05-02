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

_ARTIFACT_SCAN_CHUNK_MS = 10
_ARTIFACT_SCAN_PADDING_MS = 1200
_ARTIFACT_ENERGY_CONTEXT_SECONDS = 5


def _rms(audio: torch.Tensor, dim=None) -> torch.Tensor:
    """Return root-mean-square amplitude for an audio tensor over ``dim``."""
    return torch.sqrt(torch.mean(audio.float() ** 2, dim=dim))


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
    max_artifact_ms: int = 800,
    energy_ratio_cap: float = 0.65,
    max_passes: int = 3,
) -> torch.Tensor:
    """
    Trim energy bursts that appear after silence gaps near the end of audio.

    Masked diffusion models occasionally over-generate extra frames beyond
    the last real phoneme, producing artifact bursts (50-600ms) that sit above
    the silence threshold and survive ``remove_silence_edges()``.

    The actual pattern in the waveform is often::

        [real speech] → [silence gap] → [artifact burst] → [trailing silence]

    Multiple artifacts can stack (burst → silence → burst → silence), so the
    function runs iteratively up to ``max_passes`` until no more artifacts are
    found.

    An additional energy-ratio guard prevents trimming legitimate speech: the
    burst's RMS must be below ``energy_ratio_cap`` × the RMS of recent main-body
    audio before the trim point.

    Parameters:
        audio: PyTorch tensor of shape (C, T)
        sampling_rate: sample rate of the audio
        silence_thresh: dBFS level below which a frame is "silent" (default -50)
        min_silence_ms: minimum gap length in ms to qualify as a silence boundary
        max_artifact_ms: maximum length in ms of a post-silence burst to trim
        energy_ratio_cap: burst RMS must be below this fraction of main RMS
        max_passes: maximum number of iterative trim passes

    Returns:
        PyTorch tensor (C, T') with trailing artifacts removed (if detected)
    """
    chunk_samples = int(round(_ARTIFACT_SCAN_CHUNK_MS * sampling_rate / 1000))
    min_silence_samples = int(round(min_silence_ms * sampling_rate / 1000))
    max_artifact_samples = int(round(max_artifact_ms * sampling_rate / 1000))
    if chunk_samples <= 0 or min_silence_samples <= 0 or max_artifact_samples <= 0:
        raise ValueError("sampling_rate is too low for the configured trim windows")
    scan_samples = max_artifact_samples + max(
        min_silence_samples,
        int(round(_ARTIFACT_SCAN_PADDING_MS * sampling_rate / 1000)),
    )
    silence_rms = 10 ** (silence_thresh / 20)

    for _ in range(max_passes):
        total_samples = audio.size(-1)
        if total_samples < min_silence_samples + chunk_samples:
            break

        scan_start = max(0, total_samples - scan_samples)
        window = audio[:, scan_start:]
        num_chunks = (window.size(-1) + chunk_samples - 1) // chunk_samples
        padded_len = num_chunks * chunk_samples
        window = torch.nn.functional.pad(window, (0, padded_len - window.size(-1)))
        chunked = window.float().reshape(audio.size(0), num_chunks, chunk_samples)
        loud = _rms(chunked, dim=(0, 2)) > silence_rms

        trim_at = None
        in_burst = False
        burst_start = None
        burst_len = 0
        silence_len = 0

        for chunk_idx in range(num_chunks - 1, -1, -1):
            pos = scan_start + chunk_idx * chunk_samples
            end = min(pos + chunk_samples, total_samples)
            chunk_len = end - pos
            if not in_burst:
                if loud[chunk_idx].item():
                    in_burst = True
                    burst_start = pos
                    burst_len = chunk_len
                    silence_len = 0
            else:
                if loud[chunk_idx].item():
                    burst_len += chunk_len
                    if burst_len > max_artifact_samples:
                        break
                else:
                    silence_len += chunk_len
                    if silence_len >= min_silence_samples:
                        trim_at = burst_start
                        break

        if trim_at is None or trim_at <= 0:
            break

        trim_samples = trim_at
        energy_context_samples = min(
            trim_samples, int(sampling_rate * _ARTIFACT_ENERGY_CONTEXT_SECONDS)
        )
        main_audio = audio[:, trim_samples - energy_context_samples : trim_samples]
        artifact_audio = audio[:, trim_samples:]
        if main_audio.numel() > 0 and artifact_audio.numel() > 0:
            main_rms = _rms(main_audio).item()
            art_rms = _rms(artifact_audio).item()
            if main_rms > 1e-6 and art_rms / main_rms < energy_ratio_cap:
                audio = audio[:, :trim_samples]
            else:
                break
        else:
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
