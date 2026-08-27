"""
Fun with Tralfam.  And Bach.  And Yo Yo Ma.
"""

import pygmu2 as pg
from pathlib import Path

SRATE = 44100
pg.set_sample_rate(SRATE)


def _sec2sam(seconds):
    """convert seconds to samples"""
    return int(round(seconds * SRATE))


AUDIO_DIR = Path(__file__).parent / "audio"
BWV_FILENAME = AUDIO_DIR / "BWV1007.wav"
BWV_AUDIO = pg.WavReaderPE(BWV_FILENAME)

# One extent per chord (more or less)
EXTENTS = [
    pg.Extent(_sec2sam(0.000000), _sec2sam(4.217464)),
    pg.Extent(_sec2sam(4.217464), _sec2sam(7.261713)),
    pg.Extent(_sec2sam(7.261713), _sec2sam(10.310325)),
    pg.Extent(_sec2sam(10.310325), _sec2sam(13.629342)),
    pg.Extent(_sec2sam(13.629342), _sec2sam(17.044310)),
    pg.Extent(_sec2sam(17.044310), _sec2sam(20.245570)),
    pg.Extent(_sec2sam(20.245570), _sec2sam(23.484025)),
]


def make_tralfam(extent):
    # shift snippet to 0, tralfam it, shift it back to original time...
    source = pg.SlicePE(BWV_AUDIO, extent.start, extent.duration)
    tralfammed = pg.TralfamPE(source, normalize_peak=0.5)
    return pg.DelayPE(tralfammed, extent.start)


# Use python's "list comprehensions" to make the complete mix
mix = pg.MixPE(*[make_tralfam(extent) for extent in EXTENTS])

if __name__ == "__main__":
    pg.play_offline(mix)
