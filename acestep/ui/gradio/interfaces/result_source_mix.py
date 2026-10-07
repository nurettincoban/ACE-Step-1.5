"""Per-sample "listen with source" controls shown under results in Lego mode."""

import gradio as gr

from acestep.ui.gradio.i18n import t


def create_source_mix_controls(n: int) -> dict:
    """Create the stem-level slider, mix button and mix player for one sample.

    The group starts hidden; mode wiring reveals it when Lego mode is active.

    Args:
        n: 1-based result slot number (used in labels).

    Returns:
        Dict with ``source_mix_group``, ``source_mix_gain``, ``source_mix_btn``
        and ``source_mix_audio`` components.
    """
    # Slider and button are stacked: result columns are too narrow to share a row.
    with gr.Group(visible=False) as source_mix_group:
        source_mix_gain = gr.Slider(
            minimum=-24.0,
            maximum=12.0,
            step=0.5,
            value=0.0,
            label=t("results.source_mix_gain"),
        )
        source_mix_btn = gr.Button(
            t("results.source_mix_btn"),
            variant="secondary",
            size="sm",
        )
        source_mix_audio = gr.Audio(
            label=t("results.source_mix_audio", n=n),
            type="filepath",
            interactive=False,
        )
    return {
        "source_mix_group": source_mix_group,
        "source_mix_gain": source_mix_gain,
        "source_mix_btn": source_mix_btn,
        "source_mix_audio": source_mix_audio,
    }
