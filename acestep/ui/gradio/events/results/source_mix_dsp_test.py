"""Unit tests for the Lego stem + source mixing DSP helpers."""

import unittest

import numpy as np

from acestep.ui.gradio.events.results.source_mix_dsp import (
    mix_stem_with_source,
    place_source,
    prevent_clipping,
    region_mask,
    resample,
    stem_region,
    to_stereo,
)

SR = 1000


def _const(value: float, seconds: float) -> np.ndarray:
    """Return a constant stereo signal of ``seconds`` at the test rate."""
    return np.full((2, int(seconds * SR)), value, dtype=np.float32)


class ToStereoTests(unittest.TestCase):
    """Channel-layout conversion."""

    def test_mono_is_duplicated(self):
        """A 1-D signal becomes two identical channels."""
        out = to_stereo(np.arange(4, dtype=np.float32))
        self.assertEqual(out.shape, (2, 4))
        np.testing.assert_array_equal(out[0], out[1])

    def test_extra_channels_are_dropped(self):
        """Only the first two channels of a multichannel file are kept."""
        out = to_stereo(np.zeros((5, 6), dtype=np.float32))
        self.assertEqual(out.shape, (2, 5))


class ResampleTests(unittest.TestCase):
    """Sample-rate conversion."""

    def test_length_scales_with_rate(self):
        """44.1 kHz to 48 kHz changes the length by the rate ratio."""
        out = resample(np.zeros((2, 44100), dtype=np.float32), 44100, 48000)
        self.assertEqual(out.shape, (2, 48000))

    def test_same_rate_is_untouched(self):
        """Matching rates return the input object unchanged."""
        audio = np.zeros((2, 10), dtype=np.float32)
        self.assertIs(resample(audio, 48000, 48000), audio)


class StemRegionTests(unittest.TestCase):
    """Mapping of the stem area onto the output timeline."""

    def test_default_area_covers_everything(self):
        """start=0, end=-1 means the whole output, with no offset."""
        self.assertEqual(stem_region(0.0, -1.0, 10.0), (0.0, 0.0, float("inf")))

    def test_negative_start_shifts_the_source(self):
        """Padding before the source moves both the source and the area right."""
        offset, start, end = stem_region(-2.0, 5.0, 10.0)
        self.assertEqual((offset, start, end), (2.0, 2.0, 7.0))


class MaskAndPlacementTests(unittest.TestCase):
    """Envelope and source placement helpers."""

    def test_mask_is_zero_outside_the_region(self):
        """Samples well outside the area are silenced, inside are full gain."""
        mask = region_mask(10 * SR, SR, 2.0, 4.0)
        self.assertEqual(mask[int(1.0 * SR)], 0.0)
        self.assertEqual(mask[int(3.0 * SR)], 1.0)
        self.assertEqual(mask[int(6.0 * SR)], 0.0)

    def test_place_source_pads_and_trims(self):
        """The source lands at the offset and is cut to the buffer length."""
        placed = place_source(_const(1.0, 3.0), offset_samples=SR, length=2 * SR)
        self.assertEqual(placed[0, SR - 1], 0.0)
        self.assertEqual(placed[0, SR], 1.0)
        self.assertEqual(placed.shape, (2, 2 * SR))

    def test_prevent_clipping_only_attenuates_hot_mixes(self):
        """Quiet audio is untouched; a hot mix is scaled under the ceiling."""
        quiet = _const(0.5, 0.1)
        self.assertIs(prevent_clipping(quiet), quiet)
        self.assertLessEqual(np.max(np.abs(prevent_clipping(_const(1.5, 0.1)))), 0.98 + 1e-6)


class MixStemWithSourceTests(unittest.TestCase):
    """End-to-end mixing behaviour."""

    def test_full_area_sums_source_and_stem(self):
        """With the default area the mix is source + stem everywhere."""
        mix = mix_stem_with_source(_const(0.2, 2.0), _const(0.3, 2.0), SR, 0.0)
        np.testing.assert_allclose(mix, 0.5, atol=1e-6)

    def test_gain_db_scales_the_stem_only(self):
        """-6 dB halves the stem (approximately) and leaves the source alone."""
        mix = mix_stem_with_source(_const(0.2, 2.0), _const(0.4, 2.0), SR, -6.0206)
        np.testing.assert_allclose(mix, 0.4, atol=1e-4)

    def test_output_follows_stem_length(self):
        """A source longer than the stem is trimmed to the stem's length."""
        mix = mix_stem_with_source(_const(0.1, 3.0), _const(0.1, 2.0), SR, 0.0)
        self.assertEqual(mix.shape, (2, 2 * SR))

    def test_stem_outside_area_is_not_added(self):
        """Outside a partial stem area only the source is heard (no doubling)."""
        mix = mix_stem_with_source(
            _const(0.2, 10.0), _const(0.2, 10.0), SR, 0.0, start_seconds=2.0, end_seconds=4.0
        )
        self.assertAlmostEqual(float(mix[0, int(1.0 * SR)]), 0.2, places=5)
        self.assertAlmostEqual(float(mix[0, int(3.0 * SR)]), 0.4, places=5)


if __name__ == "__main__":
    unittest.main()
