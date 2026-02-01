from io import BufferedIOBase
from typing import Sequence

from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx
from pygmu2.meltysynth.model.instrument import Instrument
from pygmu2.meltysynth.model.parameters import SoundFontParameters
from pygmu2.meltysynth.model.preset import Preset
from pygmu2.meltysynth.model.sample_data import SoundFontSampleData
from pygmu2.meltysynth.model.sample_header import SampleHeader
from pygmu2.meltysynth.model.soundfont_info import SoundFontInfo


class SoundFont:
    def __init__(self, reader: BufferedIOBase) -> None:
        chunk_id = BinaryReaderEx.read_four_cc(reader)
        if chunk_id != "RIFF":
            raise Exception("The RIFF chunk was not found.")

        BinaryReaderEx.read_uint32(reader)

        form_type = BinaryReaderEx.read_four_cc(reader)
        if form_type != "sfbk":
            raise Exception(
                "The type of the RIFF chunk must be 'sfbk', but was '"
                + form_type
                + "'."
            )

        self._info = SoundFontInfo(reader)

        sample_data = SoundFontSampleData(reader)
        self._bits_per_sample = sample_data.bits_per_sample
        self._wave_data = sample_data.samples

        parameters = SoundFontParameters(reader)
        self._sample_headers = parameters.sample_headers
        self._presets = parameters.presets
        self._instruments = parameters.instruments

    @classmethod
    def from_file(cls, file_path: str) -> "SoundFont":
        with open(file_path, "rb") as f:
            return cls(f)

    @property
    def info(self) -> SoundFontInfo:
        return self._info

    @property
    def wave_data(self) -> Sequence[float]:
        return self._wave_data

    @property
    def sample_headers(self) -> Sequence[SampleHeader]:
        return self._sample_headers

    @property
    def presets(self) -> Sequence[Preset]:
        return self._presets

    @property
    def instruments(self) -> Sequence[Instrument]:
        return self._instruments
