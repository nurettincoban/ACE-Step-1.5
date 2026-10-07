"""Wiring for the Lego "listen with source" result controls.

Registers, per result slot, the mix button, a reset of the mix player when the
slot's audio changes, and visibility updates that follow the generation mode.
"""

import gradio as gr

from .. import results_handlers as res_h
from .context import GenerationWiringContext


def _clear_mix_player():
    """Return an update that empties a slot's mix player."""
    return gr.update(value=None)


def register_results_source_mix_handlers(context: GenerationWiringContext) -> None:
    """Register source-mix button, reset and visibility handlers for all slots.

    Args:
        context: Shared wiring context providing ``generation_section`` and
            ``results_section``.

    Returns:
        None: Handlers are registered in-place on the supplied components.
    """
    generation_section = context.generation_section
    results_section = context.results_section
    slots = range(1, res_h.SOURCE_MIX_SLOTS + 1)

    for idx in slots:
        results_section[f"source_mix_btn_{idx}"].click(
            fn=res_h.create_source_mix_for_slot,
            inputs=[
                results_section[f"generated_audio_{idx}"],
                results_section[f"source_mix_gain_{idx}"],
                results_section["batch_queue"],
                results_section["current_batch_index"],
                generation_section["src_audio"],
            ],
            outputs=[results_section[f"source_mix_audio_{idx}"]],
        )
        # A new generation or batch navigation replaces the stem; drop the stale mix.
        results_section[f"generated_audio_{idx}"].change(
            fn=_clear_mix_player,
            inputs=None,
            outputs=[results_section[f"source_mix_audio_{idx}"]],
        )

    generation_section["generation_mode"].change(
        fn=res_h.source_mix_visibility_updates,
        inputs=[generation_section["generation_mode"]],
        outputs=[results_section[f"source_mix_group_{idx}"] for idx in slots],
    )
