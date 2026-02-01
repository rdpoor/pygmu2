import io
from io import BufferedIOBase
from typing import Optional, Sequence

from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx


class SoundFontSampleData:
    def __init__(self, reader: BufferedIOBase) -> None:
        chunk_id = BinaryReaderEx.read_four_cc(reader)
        if chunk_id != "LIST":
            raise Exception("The LIST chunk was not found.")

        end = BinaryReaderEx.read_uint32(reader)
        end += reader.tell()

        list_type = BinaryReaderEx.read_four_cc(reader)
        if list_type != "sdta":
            raise Exception(
                "The type of the LIST chunk must be 'sdta', but was '"
                + list_type
                + "'."
            )

        bits_per_sample: int = 0
        samples: Optional[Sequence[float]] = None

        while reader.tell() < end:
            id = BinaryReaderEx.read_four_cc(reader)
            size = BinaryReaderEx.read_uint32(reader)

            match id:
                case "smpl":
                    bits_per_sample = 16
                    samples = BinaryReaderEx.read_int16_array_as_float_array(
                        reader, size
                    )
                case "sm24":
                    reader.seek(size, io.SEEK_CUR)
                case _:
                    raise MeltysynthError(
                        f"The INFO list contains an unknown ID '{sub_id}'."
                    )

        if samples is None:
            raise MeltysynthError("No valid sample data was found.")

        self._bits_per_sample = bits_per_sample
        self._samples = samples

    @property
    def bits_per_sample(self) -> int:
        return self._bits_per_sample

    @property
    def samples(self) -> Sequence[float]:
        return self._samples
