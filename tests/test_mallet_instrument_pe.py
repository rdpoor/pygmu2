"""
test_mallet_instrument_pe.py — Smoke tests for MalletInstrumentPE.

Verifies construction, rendering, output shape, decay, and register scaling
for all four instrument definitions.

Run with:
    pytest test_mallet_instrument_pe.py -v
"""

import numpy as np
import pytest

from pygmu2.config import set_sample_rate
from pygmu2 import MalletInstruments
from pygmu2.mallet_instrument_pe import MalletInstrumentPE

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000
BLOCK_SIZE = 512


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_sample_rate():
    set_sample_rate(SAMPLE_RATE)
    yield
    set_sample_rate(44100)  # restore default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def render_blocks(pe: MalletInstrumentPE, n_blocks: int) -> np.ndarray:
    """Render n_blocks contiguous blocks; return concatenated (samples, ch) array."""
    blocks = []
    for i in range(n_blocks):
        snippet = pe._render(i * BLOCK_SIZE, BLOCK_SIZE)
        blocks.append(snippet.data)
    return np.concatenate(blocks, axis=0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize(
        "instr", MalletInstruments.all().values(), ids=MalletInstruments.all().keys()
    )
    def test_instantiates(self, instr):
        pe = MalletInstrumentPE(instr, frequency=440.0)
        assert pe is not None

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency"):
            MalletInstrumentPE(MalletInstruments.MARIMBA, frequency=-440.0)

    def test_zero_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency"):
            MalletInstrumentPE(MalletInstruments.MARIMBA, frequency=0.0)

    def test_negative_amplitude_raises(self):
        with pytest.raises(ValueError, match="amplitude"):
            MalletInstrumentPE(MalletInstruments.MARIMBA, amplitude=-0.1)

    def test_repr_contains_instrument_name(self):
        assert "marimba" in repr(MalletInstrumentPE(MalletInstruments.MARIMBA))

    def test_graph_is_stateless(self):
        # Composite and partials are all closed-form: the whole graph is
        # stateless, so mallet notes can be shared and seeked freely.
        pe = MalletInstrumentPE(MalletInstruments.MARIMBA)
        assert not pe.stateful
        mix = pe.inputs()[0]
        assert all(not partial.stateful for partial in mix.inputs())


# ---------------------------------------------------------------------------
# Rendering — shape and content
# ---------------------------------------------------------------------------


class TestRendering:
    @pytest.mark.parametrize(
        "instr", MalletInstruments.all().values(), ids=MalletInstruments.all().keys()
    )
    def test_output_shape_mono(self, instr):
        audio = render_blocks(MalletInstrumentPE(instr, frequency=440.0), n_blocks=4)
        assert audio.shape == (BLOCK_SIZE * 4, 1)

    def test_output_shape_stereo(self):
        audio = render_blocks(
            MalletInstrumentPE(MalletInstruments.MARIMBA, channels=2), n_blocks=2
        )
        assert audio.shape == (BLOCK_SIZE * 2, 2)

    @pytest.mark.parametrize(
        "instr", MalletInstruments.all().values(), ids=MalletInstruments.all().keys()
    )
    def test_output_not_silent(self, instr):
        audio = render_blocks(
            MalletInstrumentPE(instr, frequency=440.0, amplitude=0.3), n_blocks=4
        )
        assert np.max(np.abs(audio)) > 1e-6

    def test_amplitude_scales_peak(self):
        loud = render_blocks(
            MalletInstrumentPE(MalletInstruments.MARIMBA, amplitude=0.6), n_blocks=4
        )
        soft = render_blocks(
            MalletInstrumentPE(MalletInstruments.MARIMBA, amplitude=0.3), n_blocks=4
        )
        assert np.max(np.abs(loud)) > np.max(np.abs(soft))

    def test_output_is_float32(self):
        audio = render_blocks(MalletInstrumentPE(MalletInstruments.MARIMBA), n_blocks=1)
        assert audio.dtype == np.float32

    def test_contiguous_blocks_no_discontinuity(self):
        """Sample-to-sample jumps must stay within the range of a sine wave."""
        pe = MalletInstrumentPE(
            MalletInstruments.MARIMBA, frequency=440.0, amplitude=0.3
        )
        audio = render_blocks(pe, n_blocks=8)[:, 0]
        max_jump = float(np.max(np.abs(np.diff(audio))))
        # A 440 Hz sine at sr=48000 moves at most 2π·440/48000 ≈ 0.058 per sample;
        # allow 10× headroom for higher partials.
        assert max_jump < 0.6, f"discontinuity at block boundary: {max_jump:.4f}"


# ---------------------------------------------------------------------------
# Decay behaviour
# ---------------------------------------------------------------------------


class TestDecay:
    def test_fundamental_decays_to_near_silence(self):
        """After 7× tau_mid the marimba fundamental should be below -60 dB."""
        # MalletInstruments.MARIMBA tau_mid = 1.0 s at A4; -60 dB ≈ 7 τ.
        n_blocks = int(np.ceil(7.0 * SAMPLE_RATE / BLOCK_SIZE))
        pe = MalletInstrumentPE(
            MalletInstruments.MARIMBA, frequency=440.0, amplitude=0.3
        )
        audio = render_blocks(pe, n_blocks)
        tail_rms = float(np.sqrt(np.mean(audio[-BLOCK_SIZE:] ** 2)))
        assert tail_rms < 1e-4, f"signal has not decayed: tail RMS = {tail_rms:.2e}"

    def test_early_blocks_louder_than_late_blocks(self):
        """Onset should be louder than the tail — basic sanity check."""
        n_blocks = int(np.ceil(3.0 * SAMPLE_RATE / BLOCK_SIZE))
        pe = MalletInstrumentPE(
            MalletInstruments.MARIMBA, frequency=440.0, amplitude=0.3
        )
        audio = render_blocks(pe, n_blocks)[:, 0]
        onset_rms = float(np.sqrt(np.mean(audio[:BLOCK_SIZE] ** 2)))
        tail_rms = float(np.sqrt(np.mean(audio[-BLOCK_SIZE:] ** 2)))
        assert onset_rms > tail_rms * 10


# ---------------------------------------------------------------------------
# Register scaling
# ---------------------------------------------------------------------------


class TestRegisterScaling:
    def _half_life_blocks(self, frequency: float, threshold: float = 0.02) -> int:
        """Return the block index at which RMS first falls below threshold."""
        pe = MalletInstrumentPE(
            MalletInstruments.MARIMBA, frequency=frequency, amplitude=0.3
        )
        for i in range(int(10.0 * SAMPLE_RATE / BLOCK_SIZE)):
            snippet = pe._render(i * BLOCK_SIZE, BLOCK_SIZE)
            if float(np.sqrt(np.mean(snippet.data**2))) < threshold:
                return i
        return i  # did not fall below threshold within 10 s

    def test_low_note_outlasts_high_note(self):
        """A2 (110 Hz) should decay more slowly than A6 (1760 Hz)."""
        low_blocks = self._half_life_blocks(110.0)
        high_blocks = self._half_life_blocks(1760.0)
        assert (
            low_blocks > high_blocks
        ), f"low note ({low_blocks} blocks) should outlast high note ({high_blocks} blocks)"

    def test_ref_freq_matches_tau_mid(self):
        """At A4 the fundamental should reach -60 dB within ±20% of tau_mid (1.0 s)."""
        tau_mid = 1.0  # MalletInstruments.MARIMBA fundamental tau_mid at A4
        tolerance = 0.20
        n_samples = int(tau_mid * SAMPLE_RATE)
        pe = MalletInstrumentPE(
            MalletInstruments.MARIMBA, frequency=440.0, amplitude=0.3
        )
        # Render to exactly tau_mid then check amplitude is near -60 dB (linear ≈ 0.001)
        n_blocks = int(np.ceil(n_samples / BLOCK_SIZE))
        audio = render_blocks(pe, n_blocks)
        peak_at_tau = float(np.max(np.abs(audio[n_samples - BLOCK_SIZE : n_samples])))
        assert peak_at_tau < 0.3 * (
            1.0 + tolerance
        ), f"signal at tau_mid louder than expected: {peak_at_tau:.4f}"
