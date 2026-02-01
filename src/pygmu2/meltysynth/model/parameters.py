from io import BufferedIOBase
from typing import Optional, Sequence

from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx
from pygmu2.meltysynth.model.generator import Generator
from pygmu2.meltysynth.model.instrument import Instrument
from pygmu2.meltysynth.model.instrument_info import InstrumentInfo
from pygmu2.meltysynth.model.modulator import Modulator
from pygmu2.meltysynth.model.preset import Preset
from pygmu2.meltysynth.model.preset_info import PresetInfo
from pygmu2.meltysynth.model.sample_header import SampleHeader
from pygmu2.meltysynth.model.zone import Zone
from pygmu2.meltysynth.model.zone_info import ZoneInfo


class SoundFontParameters:
    def __init__(self, reader: BufferedIOBase) -> None:
        chunk_id = BinaryReaderEx.read_four_cc(reader)
        if chunk_id != "LIST":
            raise Exception("The LIST chunk was not found.")

        end = BinaryReaderEx.read_int32(reader)
        end += reader.tell()

        list_type = BinaryReaderEx.read_four_cc(reader)
        if list_type != "pdta":
            raise Exception(
                "The type of the LIST chunk must be 'pdta', but was '"
                + list_type
                + "'."
            )

        preset_infos: Optional[Sequence[PresetInfo]] = None
        preset_bag: Optional[Sequence[ZoneInfo]] = None
        preset_generators: Optional[Sequence[Generator]] = None
        instrument_infos: Optional[Sequence[InstrumentInfo]] = None
        instrument_bag: Optional[Sequence[ZoneInfo]] = None
        instrument_generators: Optional[Sequence[Generator]] = None
        sample_headers: Optional[Sequence[SampleHeader]] = None

        while reader.tell() < end:
            id = BinaryReaderEx.read_four_cc(reader)
            size = BinaryReaderEx.read_uint32(reader)

            match id:
                case "phdr":
                    preset_infos = PresetInfo.read_from_chunk(reader, size)
                case "pbag":
                    preset_bag = ZoneInfo.read_from_chunk(reader, size)
                case "pmod":
                    Modulator.discard_data(reader, size)
                case "pgen":
                    preset_generators = Generator.read_from_chunk(reader, size)
                case "inst":
                    instrument_infos = InstrumentInfo.read_from_chunk(
                        reader, size
                    )
                case "ibag":
                    instrument_bag = ZoneInfo.read_from_chunk(reader, size)
                case "imod":
                    Modulator.discard_data(reader, size)
                case "igen":
                    instrument_generators = Generator.read_from_chunk(
                        reader, size
                    )
                case "shdr":
                    sample_headers = SampleHeader._read_from_chunk(reader, size)
                case _:
                    raise Exception(
                        "The INFO list contains an unknown ID '" + id + "'."
                    )

        if preset_infos is None:
            raise Exception("The PHDR sub-chunk was not found.")
        if preset_bag is None:
            raise Exception("The PBAG sub-chunk was not found.")
        if preset_generators is None:
            raise Exception("The PGEN sub-chunk was not found.")
        if instrument_infos is None:
            raise Exception("The INST sub-chunk was not found.")
        if instrument_bag is None:
            raise Exception("The IBAG sub-chunk was not found.")
        if instrument_generators is None:
            raise Exception("The IGEN sub-chunk was not found.")
        if sample_headers is None:
            raise Exception("The SHDR sub-chunk was not found.")

        self._sample_headers = sample_headers

        instrument_zones = Zone.create(instrument_bag, instrument_generators)
        self._instruments = Instrument._create(
            instrument_infos, instrument_zones, sample_headers
        )

        preset_zones = Zone.create(preset_bag, preset_generators)
        self._presets = Preset._create(
            preset_infos, preset_zones, self._instruments
        )

    @property
    def sample_headers(self) -> Sequence[SampleHeader]:
        return self._sample_headers

    @property
    def presets(self) -> Sequence[Preset]:
        return self._presets

    @property
    def instruments(self) -> Sequence[Instrument]:
        return self._instruments
