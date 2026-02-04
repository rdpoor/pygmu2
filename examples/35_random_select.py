"""
35_random_select.py

RandomChoicePE example: choose one source on each trigger and play it.
"""

import pygmu2 as pg
from pathlib import Path

pg.setup_logging(level="INFO")
logger = pg.get_logger(__name__)


def demo_boops_and_beeps():
    SAMPLE_RATE=44100

    weighted_boops = [
        (pg.SinePE(frequency=pg.pitch_to_freq(55), amplitude=0.3), 0.1),
        (pg.SinePE(frequency=pg.pitch_to_freq(57), amplitude=0.3), 0.4),
        (pg.SinePE(frequency=pg.pitch_to_freq(62), amplitude=0.3), 0.2),
        (pg.SinePE(frequency=pg.pitch_to_freq(64), amplitude=0.3), 0.3),
        (pg.SinePE(frequency=pg.pitch_to_freq(69), amplitude=0.3), 0.4),
        (pg.SinePE(frequency=pg.pitch_to_freq(71), amplitude=0.3), 0.4),
    ]

    inputs = [pe for (pe, _w) in weighted_boops]
    weights = [w for (_pe, w) in weighted_boops]

    # Trigger signal: short pulses at 10Hz
    trigger = pg.TriggerPE(
        source=pg.ConstantPE(1.0),
        trigger=pg.SinePE(frequency=10.0, amplitude=1.0),
        trigger_mode=pg.TriggerMode.RETRIGGER,
    )

    chooser = pg.RandomChoicePE(
        trigger=trigger,
        inputs=inputs,
        weights=weights,
        seed=1234,
        trigger_mode=pg.TriggerMode.RETRIGGER,
    )
    duration_seconds = 10
    renderer = pg.AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(
        pg.CropPE(chooser, pg.Extent(0, duration_seconds*SAMPLE_RATE)))    
    with renderer:
        renderer.start()
        renderer.play_extent()
    
    print("Done!\n", flush=True)


def demo_bongo_fury():

    pg.setup_logging(level="DEBUG")
    AUDIO_DIR = Path(__file__).parent / "audio"
    WAV_FILE = AUDIO_DIR / "djembe44.wav"
    source_stream = pg.WavReaderPE(str(WAV_FILE))
    sample_rate = source_stream.file_sample_rate or 44100

    def start_dur(start, end):
        """Convert (start, end) to (start, duration)"""
        return (start, end-start)

    slices = [
        pg.SlicePE(source_stream, *start_dur(0, 13811)),        # 0
        pg.SlicePE(source_stream, *start_dur(13811, 20882)),    # 1
        pg.SlicePE(source_stream, *start_dur(20882, 35331)),    # 2
        pg.SlicePE(source_stream, *start_dur(35331, 42732)),    # 3
        pg.SlicePE(source_stream, *start_dur(42732, 57006)),    # 4
        pg.SlicePE(source_stream, *start_dur(57006, 71456)),    # 5
        pg.SlicePE(source_stream, *start_dur(71456, 78857)),    # 6
        pg.SlicePE(source_stream, *start_dur(78857, 93130)),    # 7
        pg.SlicePE(source_stream, *start_dur(93130, 100355)),   # 8
        pg.SlicePE(source_stream, *start_dur(100355, 114541)),   # 9
    ]

    period = int(sample_rate / 10)  # triggers per second

    impulse = pg.DiracPE()
    gate = pg.TransformPE(impulse, func=lambda x: 1.0 - x)
    trigger = pg.LoopPE(gate, loop_start=0, loop_end=period)

    chooser = pg.RandomChoicePE(
        trigger=trigger,
        inputs=slices,
        seed=1234,
        trigger_mode=pg.TriggerMode.RETRIGGER,
    )
    duration_seconds = 10
    extent = pg.Extent(0, duration_seconds * sample_rate)
    logger.info(f"extent = {extent}")
    renderer = pg.AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(pg.CropPE(chooser, extent))    
    with renderer:
        renderer.start()
        renderer.play_extent()
    
    print("Done!\n", flush=True)

def demo_test_slices():

    pg.setup_logging(level="DEBUG")
    AUDIO_DIR = Path(__file__).parent / "audio"
    WAV_FILE = AUDIO_DIR / "djembe44.wav"
    source_stream = pg.WavReaderPE(str(WAV_FILE))
    sample_rate = source_stream.file_sample_rate
    logger.info(f"Sample rate = {sample_rate}")

    def start_dur(start, end):
        """Convert (start, end) to (start, duration)"""
        return (start, end-start)

    slices = [
        pg.SlicePE(source_stream, *start_dur(0, 13811)),        # 0
        pg.SlicePE(source_stream, *start_dur(13811, 20882)),    # 1
        pg.SlicePE(source_stream, *start_dur(20882, 35331)),    # 2
        pg.SlicePE(source_stream, *start_dur(35331, 42732)),    # 3
        pg.SlicePE(source_stream, *start_dur(42732, 57006)),    # 4
        pg.SlicePE(source_stream, *start_dur(57006, 71456)),    # 5
        pg.SlicePE(source_stream, *start_dur(71456, 78857)),    # 6
        pg.SlicePE(source_stream, *start_dur(78857, 93130)),    # 7
        pg.SlicePE(source_stream, *start_dur(93130, 100355)),   # 8
        pg.SlicePE(source_stream, *start_dur(100355, 114541)),   # 9
    ]

    bitz = []
    t = 0
    for s in slices:
        bitz.append(pg.DelayPE(s, t))
        t += 44100

    mix = pg.MixPE(*bitz)
    extent = mix.extent()
    logger.info(f"extent = {extent}")
    renderer = pg.AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(mix)    
    with renderer:
        renderer.start()
        renderer.play_extent()
    
    print("Done!\n", flush=True)

def demo_all():
    demo_boops_and_beeps()
    demo_bongo_fury()
    demo_test_slices()

if __name__ == "__main__":
    import sys

    demos = [
        ("1", "Demo weighted boops and beeps", demo_boops_and_beeps),
        ("2", "Demo bongo fury", demo_bongo_fury),
        ("3", "Demo test slices", demo_test_slices),
        ("a", "Demo all", demo_all),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("Demo RandomSelectPE: randomly choose one of N sources")
        print("-----------------------------------------------------")
        for key, name, _ in demos:
            print(f"  {key}: {name}")
        print()
        choice = input(f"Choice (1-{len(demos)-1} or 'a'): ").strip().lower()

    for key, _name, fn in demos:
        if key == choice:
            fn()
            break
    else:
        print("Invalid choice.")
