"""Unit tests for the Lego "listen with source" result handlers."""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from acestep.ui.gradio.events.results import source_mix


def _write_tone(path: str, seconds: float, sample_rate: int, amplitude: float) -> None:
    """Write a stereo sine tone used as a fake source or stem."""
    times = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = (amplitude * np.sin(2 * np.pi * 220 * times)).astype(np.float32)
    sf.write(path, np.stack([tone, tone], axis=1), sample_rate)


class VisibilityTests(unittest.TestCase):
    """Mode-dependent visibility of the source-mix controls."""

    def test_visible_only_in_lego_mode(self):
        """Lego shows all slots; other modes (and empty values) hide them."""
        lego = source_mix.source_mix_visibility_updates("Lego")
        self.assertEqual(len(lego), source_mix.SOURCE_MIX_SLOTS)
        self.assertTrue(all(update["visible"] for update in lego))
        for mode in ("Custom", "Remix", "Repaint", "Extract", "Complete", None):
            updates = source_mix.source_mix_visibility_updates(mode)
            self.assertFalse(any(update["visible"] for update in updates), mode)


class ResolveParamsTests(unittest.TestCase):
    """Choosing the source and stem area for a result."""

    def test_prefers_parameters_stored_with_the_batch(self):
        """The batch's own source/area win over the live Source Audio field."""
        queue = {0: {"generation_params": {
            "src_audio": "batch.wav", "repainting_start": 1.5, "repainting_end": 8.0,
        }}}
        self.assertEqual(
            source_mix.resolve_mix_params(queue, 0, "live.wav"), ("batch.wav", 1.5, 8.0)
        )

    def test_falls_back_to_live_source_with_full_area(self):
        """Without stored params the live source and a full-length area are used."""
        self.assertEqual(source_mix.resolve_mix_params({}, 0, "live.wav"), ("live.wav", 0.0, -1.0))


class CreateSourceMixTests(unittest.TestCase):
    """Click handler behaviour."""

    def setUp(self):
        """Create a temp dir and redirect mix output into it."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(source_mix, "_MIX_DIR", os.path.join(self._tmp.name, "mixes"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_renders_a_48k_mix_aligned_to_the_stem(self):
        """A 44.1 kHz source and 48 kHz stem produce a 48 kHz mix of stem length."""
        src = os.path.join(self._tmp.name, "src.mp3.wav")
        stem = os.path.join(self._tmp.name, "stem.flac")
        _write_tone(src, 2.0, 44100, 0.3)
        _write_tone(stem, 1.92, 48000, 0.3)
        update = source_mix.create_source_mix_for_slot(stem, -3.0, {}, 0, src)
        audio, rate = sf.read(update["value"])
        self.assertEqual(rate, 48000)
        self.assertEqual(audio.shape, (int(1.92 * 48000), 2))
        self.assertIn("stem_with_source", os.path.basename(update["value"]))

    @patch.object(source_mix.gr, "Warning")
    def test_missing_source_warns_without_output(self, warning):
        """A deleted source file gives a toast and leaves the player unchanged."""
        stem = os.path.join(self._tmp.name, "stem.wav")
        _write_tone(stem, 0.5, 48000, 0.3)
        update = source_mix.create_source_mix_for_slot(stem, 0.0, {}, 0, "/missing.wav")
        warning.assert_called_once()
        self.assertNotIn("value", update)

    @patch.object(source_mix.gr, "Warning")
    def test_empty_slot_warns_without_output(self, warning):
        """Clicking before any stem exists only shows a toast."""
        update = source_mix.create_source_mix_for_slot(None, 0.0, {}, 0, None)
        warning.assert_called_once()
        self.assertNotIn("value", update)


if __name__ == "__main__":
    unittest.main()
