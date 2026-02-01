import io
from io import BufferedIOBase

from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx


class Modulator:
    @staticmethod
    def discard_data(reader: BufferedIOBase, size: int) -> None:
        if size % 10 != 0:
            raise MeltysynthError("The modulator list is invalid.")
        reader.seek(size, io.SEEK_CUR)
