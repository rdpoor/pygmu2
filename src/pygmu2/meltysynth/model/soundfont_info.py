from io import BufferedIOBase

from pygmu2.meltysynth.exceptions import MeltysynthError
from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx
from pygmu2.meltysynth.model.types import SoundFontVersion


class SoundFontInfo:
    """SoundFont INFO chunk metadata (version, names, comments)."""

    def __init__(self, reader: BufferedIOBase) -> None:
        self._version = SoundFontVersion(0, 0)
        self._target_sound_engine = ""
        self._bank_name = ""
        self._rom_name = ""
        self._rom_version = SoundFontVersion(0, 0)
        self._creation_date = ""
        self._author = ""
        self._target_product = ""
        self._copyright = ""
        self._comments = ""
        self._tools = ""

        chunk_id = BinaryReaderEx.read_four_cc(reader)
        if chunk_id != "LIST":
            raise MeltysynthError("The LIST chunk was not found.")

        end = BinaryReaderEx.read_uint32(reader)
        end += reader.tell()

        list_type = BinaryReaderEx.read_four_cc(reader)
        if list_type != "INFO":
            raise MeltysynthError(
                f"The type of the LIST chunk must be 'INFO', but was '{list_type}'."
            )

        while reader.tell() < end:
            sub_id = BinaryReaderEx.read_four_cc(reader)
            size = BinaryReaderEx.read_uint32(reader)

            match sub_id:
                case "ifil":
                    self._version = SoundFontVersion(
                        BinaryReaderEx.read_uint16(reader),
                        BinaryReaderEx.read_uint16(reader),
                    )
                case "isng":
                    self._target_sound_engine = (
                        BinaryReaderEx.read_fixed_length_string(reader, size)
                    )
                case "INAM":
                    self._bank_name = BinaryReaderEx.read_fixed_length_string(
                        reader, size
                    )
                case "irom":
                    self._rom_name = BinaryReaderEx.read_fixed_length_string(
                        reader, size
                    )
                case "iver":
                    self._rom_version = SoundFontVersion(
                        BinaryReaderEx.read_uint16(reader),
                        BinaryReaderEx.read_uint16(reader),
                    )
                case "ICRD":
                    self._creation_date = (
                        BinaryReaderEx.read_fixed_length_string(reader, size)
                    )
                case "IENG":
                    self._author = BinaryReaderEx.read_fixed_length_string(
                        reader, size
                    )
                case "IPRD":
                    self._target_product = (
                        BinaryReaderEx.read_fixed_length_string(reader, size)
                    )
                case "ICOP":
                    self._copyright = (
                        BinaryReaderEx.read_fixed_length_string(reader, size)
                    )
                case "ICMT":
                    self._comments = BinaryReaderEx.read_fixed_length_string(
                        reader, size
                    )
                case "ISFT":
                    self._tools = BinaryReaderEx.read_fixed_length_string(
                        reader, size
                    )
                case _:
                    raise MeltysynthError(
                        f"The INFO list contains an unknown ID '{sub_id}'."
                    )

    @property
    def version(self) -> SoundFontVersion:
        return self._version

    @property
    def target_sound_engine(self) -> str:
        return self._target_sound_engine

    @property
    def bank_name(self) -> str:
        return self._bank_name

    @property
    def rom_name(self) -> str:
        return self._rom_name

    @property
    def rom_version(self) -> SoundFontVersion:
        return self._rom_version

    @property
    def creation_date(self) -> str:
        return self._creation_date

    @property
    def author(self) -> str:
        return self._author

    @property
    def target_product(self) -> str:
        return self._target_product

    @property
    def copyright(self) -> str:
        return self._copyright

    @property
    def comments(self) -> str:
        return self._comments

    @property
    def tools(self) -> str:
        return self._tools
