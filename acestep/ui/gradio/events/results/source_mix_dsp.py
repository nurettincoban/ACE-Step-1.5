"""Pure DSP helpers that overlay a generated Lego stem on its source audio.

All arrays are channels-first ``float32`` numpy arrays shaped ``[2, samples]``.
Nothing here touches files, Gradio, or the GPU.
"""

from math import gcd

import numpy as np
from scipy.signal import resample_poly

_FADE_SECONDS = 0.02
_CLIP_CEILING = 0.98


def to_stereo(audio: np.ndarray) -> np.ndarray:
    """Convert soundfile-layout audio to channels-first stereo.

    Args:
        audio: ``[samples]`` (mono) or ``[samples, channels]`` array.

    Returns:
        ``[2, samples]`` float32 array; mono is duplicated, extra channels dropped.
    """
    if audio.ndim == 1:
        audio = audio[:, None]
    channels_first = audio.T.astype(np.float32, copy=False)
    if channels_first.shape[0] == 1:
        channels_first = np.repeat(channels_first, 2, axis=0)
    return channels_first[:2]


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample channels-first audio with a polyphase filter.

    Args:
        audio: ``[channels, samples]`` array.
        source_rate: Sample rate of ``audio``.
        target_rate: Desired sample rate.

    Returns:
        Resampled float32 array (unchanged when the rates already match).
    """
    if source_rate == target_rate:
        return audio
    divisor = gcd(source_rate, target_rate)
    up, down = target_rate // divisor, source_rate // divisor
    return resample_poly(audio, up, down, axis=-1).astype(np.float32)


def stem_region(
    start_seconds: float, end_seconds: float, source_seconds: float
) -> tuple[float, float, float]:
    """Map the Lego stem area onto the generated output's timeline.

    A negative start pads silence before the source, which shifts everything
    right by that amount; an end of ``-1`` (or past the source) means "to the end".

    Args:
        start_seconds: ``repainting_start`` used for the generation.
        end_seconds: ``repainting_end`` used for the generation (``-1`` = end).
        source_seconds: Duration of the source audio.

    Returns:
        ``(source_offset, region_start, region_end)`` in output seconds;
        ``region_end`` is ``inf`` when the area runs to the end.
    """
    offset = max(0.0, -start_seconds)
    region_start = max(0.0, start_seconds) + offset
    open_ended = end_seconds < 0 or end_seconds >= source_seconds
    region_end = float("inf") if open_ended else end_seconds + offset
    return offset, region_start, region_end


def region_mask(
    length: int, sample_rate: int, region_start: float, region_end: float
) -> np.ndarray:
    """Build a 0..1 gain envelope that is 1 inside the region, with short fades.

    Args:
        length: Number of samples in the output.
        sample_rate: Output sample rate.
        region_start: Region start in seconds.
        region_end: Region end in seconds (``inf`` = to the end).

    Returns:
        ``[length]`` float32 envelope.
    """
    times = np.arange(length, dtype=np.float32) / sample_rate
    fade_in = np.clip((times - region_start) / _FADE_SECONDS + 1.0, 0.0, 1.0)
    if np.isinf(region_end):
        return fade_in.astype(np.float32)
    fade_out = np.clip((region_end - times) / _FADE_SECONDS + 1.0, 0.0, 1.0)
    return np.minimum(fade_in, fade_out).astype(np.float32)


def place_source(source: np.ndarray, offset_samples: int, length: int) -> np.ndarray:
    """Position the source at ``offset_samples`` inside a buffer of ``length``.

    Args:
        source: ``[2, samples]`` source audio.
        offset_samples: Leading silence before the source starts.
        length: Length of the returned buffer.

    Returns:
        ``[2, length]`` array, zero-padded or trimmed as needed.
    """
    placed = np.zeros((source.shape[0], length), dtype=np.float32)
    usable = max(0, min(source.shape[1], length - offset_samples))
    placed[:, offset_samples:offset_samples + usable] = source[:, :usable]
    return placed


def prevent_clipping(mix: np.ndarray) -> np.ndarray:
    """Scale the mix down uniformly when its peak exceeds the ceiling.

    Args:
        mix: ``[channels, samples]`` audio.

    Returns:
        The mix, attenuated only if it would clip.
    """
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak <= _CLIP_CEILING:
        return mix
    return (mix * (_CLIP_CEILING / peak)).astype(np.float32)


def mix_stem_with_source(
    source: np.ndarray,
    stem: np.ndarray,
    sample_rate: int,
    stem_gain_db: float,
    start_seconds: float = 0.0,
    end_seconds: float = -1.0,
) -> np.ndarray:
    """Overlay the generated stem on the source, aligned to the stem's timeline.

    Args:
        source: ``[2, samples]`` source audio at ``sample_rate``.
        stem: ``[2, samples]`` generated stem at ``sample_rate``.
        sample_rate: Shared sample rate of both inputs.
        stem_gain_db: Gain applied to the stem before summing.
        start_seconds: Stem area start used for the generation.
        end_seconds: Stem area end used for the generation (``-1`` = end).

    Returns:
        ``[2, len(stem)]`` mix that never exceeds the clipping ceiling.
    """
    length = stem.shape[1]
    offset, region_start, region_end = stem_region(
        start_seconds, end_seconds, source.shape[1] / sample_rate
    )
    placed = place_source(source, int(round(offset * sample_rate)), length)
    envelope = region_mask(length, sample_rate, region_start, region_end)
    gain = float(10.0 ** (stem_gain_db / 20.0))
    return prevent_clipping(placed + stem * envelope * gain)
