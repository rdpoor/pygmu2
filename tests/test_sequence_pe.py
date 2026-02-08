import numpy as np

import pygmu2 as pg


def _seg(val: float, dur: int) -> pg.ProcessingElement:
    return pg.CropPE(pg.ConstantPE(val), 0, dur)


def test_sequence_overlap_mix():
    sample_rate = 10
    pg.set_sample_rate(sample_rate)
    a = _seg(1.0, 10)
    b = _seg(2.0, 5)

    seq = pg.SequencePE(
        (a, 0),
        (b, 5),
        mode=pg.SequenceMode.OVERLAP,
    )
    out = seq.render(0, 10).data[:, 0]
    expected = np.array([1, 1, 1, 1, 1, 3, 3, 3, 3, 3], dtype=np.float32)

    assert np.allclose(out, expected)


def test_sequence_cut_stops_previous():
    sample_rate = 10
    pg.set_sample_rate(sample_rate)
    a = _seg(1.0, 5)
    b = _seg(2.0, 5)

    seq = pg.SequencePE(
        (a, 0),
        (b, 5),
        mode=pg.SequenceMode.NON_OVERLAP,
    )
    out = seq.render(0, 10).data[:, 0]
    expected = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.float32)

    assert np.allclose(out, expected)


def test_sequence_auto_start_none():
    sample_rate = 10
    pg.set_sample_rate(sample_rate)
    a = pg.SlicePE(pg.IdentityPE(), start=0, duration=5)
    b = pg.SlicePE(pg.IdentityPE(), start=10, duration=5)

    seq = pg.SequencePE(
        (a, None),  # auto -> 0
        (b, None),  # auto -> end of a (5)
        mode=pg.SequenceMode.NON_OVERLAP,
    )
    out = seq.render(0, 10).data[:, 0]
    expected = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14], dtype=np.float32)

    assert np.allclose(out, expected)


def test_sequence_auto_start_none_after_infinite_extent_raises():
    a = pg.IdentityPE()  # infinite extent
    b = pg.SinePE()

    try:
        pg.SequencePE(
            (a, 0),
            (b, None),
            mode=pg.SequenceMode.NON_OVERLAP,
        )
    except ValueError as exc:
        assert "infinite extent" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for auto-advance after infinite extent")


def test_sequence_auto_start_none_overlap_mode():
    sample_rate = 10
    pg.set_sample_rate(sample_rate)
    a = _seg(1.0, 3)
    b = _seg(2.0, 3)

    seq = pg.SequencePE(
        (a, None),
        (b, None),
        mode=pg.SequenceMode.OVERLAP,
    )
    out = seq.render(0, 6).data[:, 0]
    expected = np.array([1, 1, 1, 2, 2, 2], dtype=np.float32)
    assert np.allclose(out, expected)


def test_sequence_multiple_consecutive_none():
    sample_rate = 10
    pg.set_sample_rate(sample_rate)
    a = _seg(1.0, 2)
    b = _seg(2.0, 2)
    c = _seg(3.0, 2)

    seq = pg.SequencePE(
        (a, None),
        (b, None),
        (c, None),
        mode=pg.SequenceMode.NON_OVERLAP,
    )
    out = seq.render(0, 6).data[:, 0]
    expected = np.array([1, 1, 2, 2, 3, 3], dtype=np.float32)
    assert np.allclose(out, expected)


def test_sequence_mixed_explicit_and_none():
    sample_rate = 10
    pg.set_sample_rate(sample_rate)
    a = _seg(1.0, 2)  # starts at 0
    b = _seg(2.0, 2)  # explicit start at 5
    c = _seg(3.0, 2)  # auto start after b

    seq = pg.SequencePE(
        (a, 0),
        (b, 5),
        (c, None),
        mode=pg.SequenceMode.NON_OVERLAP,
    )
    out = seq.render(0, 9).data[:, 0]
    expected = np.array([1, 1, 0, 0, 0, 2, 2, 3, 3], dtype=np.float32)
    assert np.allclose(out, expected)


def test_sequence_zero_duration_segment():
    sample_rate = 10
    pg.set_sample_rate(sample_rate)
    a = _seg(1.0, 0)  # zero-duration
    b = _seg(2.0, 2)

    seq = pg.SequencePE(
        (a, None),
        (b, None),  # should start at 0
        mode=pg.SequenceMode.NON_OVERLAP,
    )
    out = seq.render(0, 2).data[:, 0]
    expected = np.array([2, 2], dtype=np.float32)
    assert np.allclose(out, expected)


def test_sequence_out_of_order_starts_sorted_after_normalization():
    sample_rate = 10
    pg.set_sample_rate(sample_rate)
    a = _seg(1.0, 2)  # explicit start 10
    b = _seg(2.0, 2)  # auto start after a -> 12
    c = _seg(3.0, 2)  # explicit start 0

    seq = pg.SequencePE(
        (a, 10),
        (b, None),
        (c, 0),
        mode=pg.SequenceMode.NON_OVERLAP,
    )
    out = seq.render(0, 14).data[:, 0]
    expected = np.array(
        [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2],
        dtype=np.float32,
    )
    assert np.allclose(out, expected)


def test_sequence_negative_start_time():
    sample_rate = 10
    pg.set_sample_rate(sample_rate)
    a = _seg(1.0, 2)  # start at -2, so it ends at 0
    b = _seg(2.0, 2)  # auto start at 0

    seq = pg.SequencePE(
        (a, -2),
        (b, None),
        mode=pg.SequenceMode.NON_OVERLAP,
    )
    out = seq.render(0, 2).data[:, 0]
    expected = np.array([2, 2], dtype=np.float32)
    assert np.allclose(out, expected)
