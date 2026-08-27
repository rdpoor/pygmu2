"""
test_decaying_sine_pe.py — Tests for DecayingSinePE.

Reflects the tau refactor:
  - tau  replaces the old duration / rho pair
  - db_floor (default −60) sets the crop point; extent is now finite
  - rho_for_decay_db static method removed

Run with:  pytest test_decaying_sine_pe.py -v
"""

import math

import numpy as np
import pytest

from pygmu2.config import set_sample_rate
from pygmu2.decaying_sine_pe import DecayingSinePE

SAMPLE_RATE = 48000
BLOCK_SIZE = 512


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_sample_rate():
    set_sample_rate(SAMPLE_RATE)
    yield
    set_sample_rate(44100)  # restore default


def render_blocks(pe: DecayingSinePE, n_blocks: int) -> np.ndarray:
    """Render n_blocks contiguous blocks; return (samples, channels) array."""
    blocks = []
    for i in range(n_blocks):
        snippet = pe._render(i * BLOCK_SIZE, BLOCK_SIZE)
        blocks.append(snippet.data)
    return np.concatenate(blocks, axis=0)


def expected_crop_samples(tau: float, db_floor: float = -60.0) -> int:
    """Mirror the PE's internal crop formula for independent verification."""
    return math.ceil(-tau * SAMPLE_RATE * (db_floor / 20.0) * math.log(10))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_valid_instantiation(self):
        assert DecayingSinePE(frequency=440.0, tau=0.5) is not None

    def test_zero_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency"):
            DecayingSinePE(frequency=0.0, tau=0.5)

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency"):
            DecayingSinePE(frequency=-440.0, tau=0.5)

    def test_zero_amplitude_raises(self):
        with pytest.raises(ValueError, match="amplitude"):
            DecayingSinePE(frequency=440.0, amplitude=0.0, tau=0.5)

    def test_negative_amplitude_raises(self):
        with pytest.raises(ValueError, match="amplitude"):
            DecayingSinePE(frequency=440.0, amplitude=-0.3, tau=0.5)

    def test_zero_tau_raises(self):
        with pytest.raises(ValueError, match="tau"):
            DecayingSinePE(frequency=440.0, tau=0.0)

    def test_negative_tau_raises(self):
        with pytest.raises(ValueError, match="tau"):
            DecayingSinePE(frequency=440.0, tau=-1.0)

    def test_zero_db_floor_raises(self):
        with pytest.raises(ValueError, match="db_floor"):
            DecayingSinePE(frequency=440.0, tau=0.5, db_floor=0.0)

    def test_positive_db_floor_raises(self):
        with pytest.raises(ValueError, match="db_floor"):
            DecayingSinePE(frequency=440.0, tau=0.5, db_floor=6.0)

    def test_rho_parameter_removed(self):
        """rho was an accepted parameter before the tau refactor."""
        with pytest.raises(TypeError):
            DecayingSinePE(frequency=440.0, rho=0.9999)

    def test_duration_parameter_removed(self):
        """duration was an accepted parameter before the tau refactor."""
        with pytest.raises(TypeError):
            DecayingSinePE(frequency=440.0, duration=2.0)

    def test_rho_for_decay_db_removed(self):
        """Static helper was removed along with the rho parameter."""
        assert not hasattr(DecayingSinePE, "rho_for_decay_db")

    def test_stateless(self):
        # Closed-form generation: random access, no state to corrupt.
        assert not DecayingSinePE(frequency=440.0, tau=0.5).stateful


# ---------------------------------------------------------------------------
# Rendering — shape, type, basic content
# ---------------------------------------------------------------------------


class TestRendering:
    def test_output_shape_mono(self):
        audio = render_blocks(DecayingSinePE(frequency=440.0, tau=0.5), n_blocks=4)
        assert audio.shape == (BLOCK_SIZE * 4, 1)

    def test_output_shape_stereo(self):
        audio = render_blocks(
            DecayingSinePE(frequency=440.0, tau=0.5, channels=2), n_blocks=2
        )
        assert audio.shape == (BLOCK_SIZE * 2, 2)

    def test_output_is_float32(self):
        audio = render_blocks(DecayingSinePE(frequency=440.0, tau=0.5), n_blocks=1)
        assert audio.dtype == np.float32

    def test_onset_is_not_silent(self):
        audio = render_blocks(
            DecayingSinePE(frequency=440.0, tau=0.5, amplitude=0.3), n_blocks=1
        )
        assert np.max(np.abs(audio)) > 1e-4

    def test_amplitude_scales_output(self):
        loud = render_blocks(
            DecayingSinePE(frequency=440.0, tau=0.5, amplitude=0.6), n_blocks=2
        )
        soft = render_blocks(
            DecayingSinePE(frequency=440.0, tau=0.5, amplitude=0.3), n_blocks=2
        )
        assert np.max(np.abs(loud)) > np.max(np.abs(soft))

    def test_stereo_channels_are_identical(self):
        audio = render_blocks(
            DecayingSinePE(frequency=440.0, tau=0.5, channels=2), n_blocks=4
        )
        np.testing.assert_array_equal(audio[:, 0], audio[:, 1])

    def test_contiguous_blocks_no_discontinuity(self):
        """Sample-to-sample jump must stay within the range of the waveform."""
        pe = DecayingSinePE(frequency=440.0, tau=0.5, amplitude=0.3)
        audio = render_blocks(pe, n_blocks=8)[:, 0]
        max_jump = float(np.max(np.abs(np.diff(audio))))
        # 440 Hz at 48 kHz moves ≤ 2π·440/48000 ≈ 0.058 per sample; 10× headroom
        assert max_jump < 0.6, f"discontinuity detected: {max_jump:.4f}"


# ---------------------------------------------------------------------------
# Decay envelope
# ---------------------------------------------------------------------------


class TestDecayEnvelope:
    def _rms_near(
        self, audio: np.ndarray, center: int, half_window: int = 256
    ) -> float:
        """RMS of channel 0 in a window of ±half_window samples around center."""
        lo = max(0, center - half_window)
        hi = min(audio.shape[0], center + half_window)
        chunk = audio[lo:hi, 0]
        return float(np.sqrt(np.mean(chunk**2))) if len(chunk) else 0.0

    def test_signal_decays_over_time(self):
        tau = 0.2
        n_tau = int(tau * SAMPLE_RATE)
        n_blocks = math.ceil((n_tau + BLOCK_SIZE) / BLOCK_SIZE)
        pe = DecayingSinePE(frequency=440.0, tau=tau, amplitude=0.3)
        audio = render_blocks(pe, n_blocks)
        assert self._rms_near(audio, BLOCK_SIZE // 2) > self._rms_near(audio, n_tau) * 2

    def test_envelope_at_one_time_constant(self):
        """
        At n = tau · sr, envelope = A · exp(−1) ≈ 0.368 · A.
        For a sine, RMS ≈ envelope / √2.  Allow ±20 % for windowing.
        """
        tau, amplitude = 0.2, 0.5
        n_tau = int(tau * SAMPLE_RATE)
        n_blocks = math.ceil((n_tau + BLOCK_SIZE) / BLOCK_SIZE)
        # 880 Hz gives ~55 samples/cycle so a 512-sample window spans ~9 cycles.
        pe = DecayingSinePE(frequency=880.0, tau=tau, amplitude=amplitude)
        audio = render_blocks(pe, n_blocks)
        measured = self._rms_near(audio, n_tau, half_window=512)
        expected = amplitude * math.exp(-1) / math.sqrt(2)
        assert (
            abs(measured - expected) / expected < 0.20
        ), f"envelope at tau: expected RMS {expected:.4f}, got {measured:.4f}"

    def test_larger_tau_decays_more_slowly(self):
        sample_offset = int(0.1 * SAMPLE_RATE)
        n_blocks = math.ceil((sample_offset + BLOCK_SIZE) / BLOCK_SIZE)
        fast = render_blocks(
            DecayingSinePE(frequency=440.0, tau=0.1, amplitude=0.3), n_blocks
        )
        slow = render_blocks(
            DecayingSinePE(frequency=440.0, tau=0.2, amplitude=0.3), n_blocks
        )
        assert self._rms_near(slow, sample_offset) > self._rms_near(fast, sample_offset)


# ---------------------------------------------------------------------------
# Crop and extent
# ---------------------------------------------------------------------------


class TestCropAndExtent:
    def test_extent_is_finite_after_first_render(self):
        pe = DecayingSinePE(frequency=440.0, tau=0.1)
        pe._render(0, BLOCK_SIZE)
        extent = pe._compute_extent()
        assert extent.end is not None

    def test_extent_start_is_zero(self):
        pe = DecayingSinePE(frequency=440.0, tau=0.1)
        pe._render(0, BLOCK_SIZE)
        assert pe._compute_extent().start == 0

    def test_extent_end_matches_crop_formula(self):
        tau, db_floor = 0.1, -60.0
        pe = DecayingSinePE(frequency=440.0, tau=tau, db_floor=db_floor)
        pe._render(0, BLOCK_SIZE)
        assert pe._compute_extent().end == expected_crop_samples(tau, db_floor)

    def test_silence_past_crop_point(self):
        """A block starting entirely after crop_samples must be all zeros."""
        tau, db_floor = 0.05, -60.0
        crop = expected_crop_samples(tau, db_floor)
        pe = DecayingSinePE(frequency=440.0, tau=tau, db_floor=db_floor)
        pe._render(0, BLOCK_SIZE)  # warm up
        snippet = pe._render(crop + BLOCK_SIZE, BLOCK_SIZE)
        assert np.all(snippet.data == 0.0)

    def test_shallower_db_floor_crops_earlier(self):
        assert expected_crop_samples(0.2, -30.0) < expected_crop_samples(0.2, -60.0)

    def test_deeper_db_floor_crops_later(self):
        assert expected_crop_samples(0.2, -90.0) > expected_crop_samples(0.2, -60.0)

    def test_crop_samples_proportional_to_tau(self):
        """Doubling tau should double crop_samples within rounding."""
        short = expected_crop_samples(0.1)
        long = expected_crop_samples(0.2)
        assert abs(long - 2 * short) <= 2

    def test_extent_is_finite_before_render(self):
        """crop_samples is known at construction; extent is finite immediately."""
        pe = DecayingSinePE(frequency=440.0, tau=0.1)
        assert pe._compute_extent().end is not None

    def test_extent_survives_reset(self):
        """Extent depends only on tau and db_floor."""
        pe = DecayingSinePE(frequency=440.0, tau=0.1)
        pe._render(0, BLOCK_SIZE)
        pe.reset_state()  # no-op for a stateless PE
        assert pe._compute_extent().end == expected_crop_samples(0.1)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_frequency(self):
        assert "440.0" in repr(DecayingSinePE(frequency=440.0, tau=0.5))

    def test_repr_contains_tau(self):
        assert "tau=0.5" in repr(DecayingSinePE(frequency=440.0, tau=0.5))

    def test_repr_omits_db_floor_at_default(self):
        assert "db_floor" not in repr(DecayingSinePE(frequency=440.0, tau=0.5))

    def test_repr_includes_db_floor_when_nondefault(self):
        assert "db_floor=-40.0" in repr(
            DecayingSinePE(frequency=440.0, tau=0.5, db_floor=-40.0)
        )
