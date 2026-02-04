import numpy as np

import pygmu2 as pg


def test_sequence_overlap_mix():
    sample_rate = 10
    a = pg.ConstantPE(1.0)
    b = pg.ConstantPE(2.0)

    seq = pg.SequencePE(
        (a, 0),
        (b, 5),
        mode=pg.SequenceMode.OVERLAP,
    )
    seq.configure(sample_rate)

    out = seq.render(0, 10).data[:, 0]
    expected = np.array([1, 1, 1, 1, 1, 3, 3, 3, 3, 3], dtype=np.float32)

    assert np.allclose(out, expected)


def test_sequence_cut_stops_previous():
    sample_rate = 10
    a = pg.ConstantPE(1.0)
    b = pg.ConstantPE(2.0)

    seq = pg.SequencePE(
        (a, 0),
        (b, 5),
        mode=pg.SequenceMode.NON_OVERLAP,
    )
    seq.configure(sample_rate)

    out = seq.render(0, 10).data[:, 0]
    expected = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.float32)

    assert np.allclose(out, expected)
