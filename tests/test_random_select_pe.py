import numpy as np

import pygmu2 as pg


def test_random_select_retrigger_resets_on_trigger():
    sample_rate = 10  # small for easy reasoning
    pg.set_sample_rate(sample_rate)

    source = pg.IdentityPE()
    slice_a = pg.SlicePE(source, start=0, duration=5)
    slice_b = pg.SlicePE(source, start=3, duration=5)

    trigger = pg.FunctionGenPE(
        frequency=1.0,  # 1 Hz => 10-sample period at sr=10
        duty_cycle=0.5,  # 5 samples high, 5 samples low
        waveform="rectangle",
        channels=1,
    )

    chooser = pg.RandomSelectPE(
        trigger=trigger,
        inputs=[slice_a, slice_b],
        weights=[0.0, 1.0],  # always choose slice_b
        seed=1234,
        trigger_mode=pg.TriggerMode.RETRIGGER,
    )

    snippet = chooser.render(0, 20)
    out = snippet.data[:, 0]

    expected = np.array(
        [3, 4, 5, 6, 7, 0, 0, 0, 0, 0,
         3, 4, 5, 6, 7, 0, 0, 0, 0, 0],
        dtype=np.float32,
    )

    assert np.allclose(out, expected)


def test_random_select_dirac_low_sample_retrigger():
    sample_rate = 10  # small for easy reasoning
    period = sample_rate  # 1 Hz
    pg.set_sample_rate(sample_rate)

    source = pg.IdentityPE()
    slice_a = pg.SlicePE(source, start=0, duration=5)
    slice_b = pg.SlicePE(source, start=3, duration=5)

    impulse = pg.DiracPE()
    gate = pg.TransformPE(impulse, func=lambda x: 1.0 - x)
    trigger = pg.LoopPE(gate, loop_start=0, loop_end=period)

    chooser = pg.RandomSelectPE(
        trigger=trigger,
        inputs=[slice_a, slice_b],
        weights=[0.0, 1.0],  # always choose slice_b
        seed=1234,
        trigger_mode=pg.TriggerMode.RETRIGGER,
    )

    snippet = chooser.render(0, 20)
    out = snippet.data[:, 0]

    expected = np.array(
        [0, 3, 4, 5, 6, 7, 0, 0, 0, 0,
         0, 3, 4, 5, 6, 7, 0, 0, 0, 0],
        dtype=np.float32,
    )

    assert np.allclose(out, expected)

def test_random_select_verify_trigger():
    sample_rate = 44100
    pg.set_sample_rate(sample_rate)
    source_stream = pg.IdentityPE()

    slices = [
        pg.SlicePE(source_stream, 10, 15),  # start at 10, end at 15, dur = 5
    ]

    impulse = pg.DiracPE()
    gate = pg.TransformPE(impulse, func=lambda x: 1.0 - x)
    trigger = pg.LoopPE(gate, loop_start=0, loop_end=10)  # trigger every 10

    snippet = trigger.render(0, 20)
    out = snippet.data[:, 0]
    expected = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        ])
    assert np.allclose(out, expected)

def test_random_select_slice_shorter_than_retrigger():
    sample_rate = 44100
    pg.set_sample_rate(sample_rate)

    slices = [
        pg.SlicePE(pg.IdentityPE(), 10, 5),  # start at 10, end before 15, dur = 5
    ]

    trigger = pg.ArrayPE([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    ])

    chooser = pg.RandomSelectPE(
        trigger=trigger,
        inputs=slices,
        seed=1234,
        trigger_mode=pg.TriggerMode.RETRIGGER,
    )

    snippet = chooser.render(0, 20)
    out = snippet.data[:, 0]
    expected = np.array([
        0, 10, 11, 12, 13, 14, 0, 0, 0, 0,
        0, 10, 11, 12, 13, 14, 0, 0, 0, 0,
    ], dtype=np.float32)
    assert np.allclose(out, expected)

def test_random_select_slice_longer_than_retrigger():
    sample_rate = 44100
    pg.set_sample_rate(sample_rate)

    slices = [
        pg.SlicePE(pg.IdentityPE(), 10, 15),  # start at 10, end before 25, dur = 15
    ]

    trigger = pg.ArrayPE([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    ])

    chooser = pg.RandomSelectPE(
        trigger=trigger,
        inputs=slices,
        seed=1234,
        trigger_mode=pg.TriggerMode.RETRIGGER,
    )

    snippet = chooser.render(0, 20)
    out = snippet.data[:, 0]
    expected = np.array([
        0, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        0, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    ], dtype=np.float32)
    assert np.allclose(out, expected)

def test_random_select_crop():
    sample_rate = 44100
    pg.set_sample_rate(sample_rate)

    cropped = pg.SlicePE(pg.IdentityPE(), 10, 5)  # start at 10, end before 15, dur = 5

    snippet = cropped.render(0, 10)     # render from 0 to 10
    out = snippet.data[:, 0]
    expected = np.array([
        10, 11, 12, 13, 14, 0, 0, 0, 0, 0,
    ], dtype=np.float32)
    assert np.allclose(out, expected)
