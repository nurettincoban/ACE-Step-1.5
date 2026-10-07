"""Result-slot handlers for listening to a Lego stem together with its source.

The mix is rendered on demand from the generation parameters stored in the
batch queue, so the main generation pipeline is left untouched.
"""

import os
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import soundfile as sf
from loguru import logger

from acestep.constants import MODE_TO_TASK_TYPE
from acestep.core.generation.handler.io_audio import _read_audio_file
from acestep.ui.gradio.events.results.generation_info import DEFAULT_RESULTS_DIR
from acestep.ui.gradio.events.results.source_mix_dsp import (
    mix_stem_with_source,
    resample,
    to_stereo,
)
from acestep.ui.gradio.i18n import t

MIX_SAMPLE_RATE = 48000
SOURCE_MIX_SLOTS = 8
_MIX_DIR = os.path.join(DEFAULT_RESULTS_DIR, "source_mixes")


def is_lego_mode(generation_mode: Optional[str]) -> bool:
    """Return whether the UI generation mode produces a Lego stem."""
    return MODE_TO_TASK_TYPE.get(generation_mode or "") == "lego"


def source_mix_visibility_updates(generation_mode: Optional[str]) -> tuple:
    """Show the per-slot source-mix controls only in Lego mode.

    Args:
        generation_mode: Current value of the generation-mode radio.

    Returns:
        One ``gr.update(visible=...)`` per result slot.
    """
    visible = is_lego_mode(generation_mode)
    return tuple(gr.update(visible=visible) for _ in range(SOURCE_MIX_SLOTS))


def resolve_mix_params(
    batch_queue: Any, batch_index: Any, fallback_src: Optional[str]
) -> tuple[Optional[str], float, float]:
    """Return the source path and stem area used to generate the current batch.

    Prefers the parameters stored with the batch (the Source Audio field may have
    changed since), falling back to the live Source Audio with a full-length area.

    Args:
        batch_queue: Batch queue state (``{index: {"generation_params": {...}}}``).
        batch_index: Index of the batch currently shown.
        fallback_src: Current Source Audio value.

    Returns:
        ``(src_path, start_seconds, end_seconds)``.
    """
    batch = (batch_queue or {}).get(batch_index) or {}
    params = batch.get("generation_params") or {}
    src_path = params.get("src_audio") or fallback_src
    start = float(params.get("repainting_start") or 0.0)
    end = params.get("repainting_end")
    return src_path, start, float(-1.0 if end is None else end)


def load_stereo_48k(path: str):
    """Read an audio file as channels-first stereo at the mix sample rate."""
    audio, sample_rate = _read_audio_file(path)
    return resample(to_stereo(audio), int(sample_rate), MIX_SAMPLE_RATE)


def mix_output_path(stem_path: str, stem_gain_db: float) -> str:
    """Build the output path for a stem's source mix (one file per gain)."""
    gain_tag = f"{stem_gain_db:+.1f}dB".replace(".", "p")
    return os.path.join(_MIX_DIR, f"{Path(stem_path).stem}_with_source_{gain_tag}.wav")


def render_source_mix(
    stem_path: str,
    src_path: str,
    stem_gain_db: float,
    start_seconds: float,
    end_seconds: float,
) -> str:
    """Render the stem over its source and write it as a 48 kHz WAV.

    Returns:
        Path of the written mix file.
    """
    mix = mix_stem_with_source(
        load_stereo_48k(src_path),
        load_stereo_48k(stem_path),
        MIX_SAMPLE_RATE,
        stem_gain_db,
        start_seconds,
        end_seconds,
    )
    output_path = mix_output_path(stem_path, stem_gain_db)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, mix.T, MIX_SAMPLE_RATE, subtype="PCM_16")
    return output_path


def create_source_mix_for_slot(
    stem_path: Optional[str],
    stem_gain_db: float,
    batch_queue: Any,
    batch_index: Any,
    fallback_src: Optional[str],
):
    """Click handler: render the slot's stem with its source for playback.

    Returns:
        ``gr.update`` for the slot's mix player (unchanged on failure, with a
        toast explaining why).
    """
    if not stem_path:
        gr.Warning(t("results.source_mix_no_stem"))
        return gr.update()
    src_path, start, end = resolve_mix_params(batch_queue, batch_index, fallback_src)
    if not src_path or not os.path.exists(src_path):
        gr.Warning(t("results.source_mix_no_source"))
        return gr.update()
    try:
        output_path = render_source_mix(stem_path, src_path, stem_gain_db, start, end)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.exception("[source_mix] Failed to render stem + source mix")
        gr.Warning(t("results.source_mix_failed", error=str(exc)))
        return gr.update()
    return gr.update(value=output_path)
