class SynthesizerSettings:
    _DEFAULT_BLOCK_SIZE = 64
    _DEFAULT_MAXIMUM_POLYPHONY = 64
    _DEFAULT_ENABLE_REVERB_AND_CHORUS = True

    def __init__(self, sample_rate: int) -> None:
        SynthesizerSettings._check_sample_rate(sample_rate)
        self._sample_rate = sample_rate
        self._block_size = SynthesizerSettings._DEFAULT_BLOCK_SIZE
        self._maximum_polyphony = SynthesizerSettings._DEFAULT_MAXIMUM_POLYPHONY
        self._enable_reverb_and_chorus = (
            SynthesizerSettings._DEFAULT_ENABLE_REVERB_AND_CHORUS
        )

    @staticmethod
    def _check_sample_rate(value: int) -> None:
        if not (16000 <= value and value <= 192000):
            raise Exception(
                "The sample rate must be between 16000 and 192000."
            )

    @staticmethod
    def _check_block_size(value: int) -> None:
        if not (8 <= value and value <= 1024):
            raise Exception("The block size must be between 8 and 1024.")

    @staticmethod
    def _check_maximum_polyphony(value: int) -> None:
        if not (8 <= value and value <= 256):
            raise Exception(
                "The maximum number of polyphony must be between 8 and 256."
            )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        SynthesizerSettings._check_sample_rate(value)
        self._sample_rate = value

    @property
    def block_size(self) -> int:
        return self._block_size

    @block_size.setter
    def block_size(self, value: int) -> None:
        SynthesizerSettings._check_block_size(value)
        self._block_size = value

    @property
    def maximum_polyphony(self) -> int:
        return self._maximum_polyphony

    @maximum_polyphony.setter
    def maximum_polyphony(self, value: int) -> None:
        SynthesizerSettings._check_maximum_polyphony(value)
        self._maximum_polyphony = value

    @property
    def enable_reverb_and_chorus(self) -> bool:
        return self._enable_reverb_and_chorus

    @enable_reverb_and_chorus.setter
    def enable_reverb_and_chorus(self, value: bool) -> None:
        self._enable_reverb_and_chorus = value
