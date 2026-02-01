from enum import IntEnum
from typing import TYPE_CHECKING

from pygmu2.meltysynth.math_utils import SoundFontMath

if TYPE_CHECKING:
    from pygmu2.meltysynth.synth.synthesizer import Synthesizer


class EnvelopeStage(IntEnum):
    DELAY = 0
    ATTACK = 1
    HOLD = 2
    DECAY = 3
    RELEASE = 4


class VolumeEnvelope:
    def __init__(self, synthesizer: "Synthesizer") -> None:
        self._synthesizer = synthesizer

    def start(
        self,
        delay: float,
        attack: float,
        hold: float,
        decay: float,
        sustain: float,
        release: float,
    ) -> None:
        self._attack_slope = 1 / attack
        self._decay_slope = -9.226 / decay
        self._release_slope = -9.226 / release

        self._attack_start_time = delay
        self._hold_start_time = self._attack_start_time + attack
        self._decay_start_time = self._hold_start_time + hold
        self._release_start_time = 0

        self._sustain_level = SoundFontMath.clamp(sustain, 0, 1)
        self._release_level = 0

        self._processed_sample_count = 0
        self._stage = EnvelopeStage.DELAY
        self._value = 0

        self.process(0)

    def release(self) -> None:
        self._stage = EnvelopeStage.RELEASE
        self._release_start_time = (
            float(self._processed_sample_count)
            / self._synthesizer.sample_rate
        )
        self._release_level = self._value

    def process(self, sample_count: int) -> bool:
        self._processed_sample_count += sample_count

        current_time = (
            float(self._processed_sample_count)
            / self._synthesizer.sample_rate
        )

        while self._stage <= EnvelopeStage.HOLD:
            end_time: float

            match self._stage:
                case EnvelopeStage.DELAY:
                    end_time = self._attack_start_time
                case EnvelopeStage.ATTACK:
                    end_time = self._hold_start_time
                case EnvelopeStage.HOLD:
                    end_time = self._decay_start_time
                case _:
                    raise ValueError("Invalid envelope stage.")

            if current_time < end_time:
                break
            else:
                self._stage = EnvelopeStage(self._stage.value + 1)

        match self._stage:
            case EnvelopeStage.DELAY:
                self._value = 0
                self._priority = 4 + self._value
                return True
            case EnvelopeStage.ATTACK:
                self._value = self._attack_slope * (
                    current_time - self._attack_start_time
                )
                self._priority = 3 + self._value
                return True
            case EnvelopeStage.HOLD:
                self._value = 1
                self._priority = 2 + self._value
                return True
            case EnvelopeStage.DECAY:
                self._value = max(
                    SoundFontMath.exp_cutoff(
                        self._decay_slope
                        * (current_time - self._decay_start_time)
                    ),
                    self._sustain_level,
                )
                self._priority = 1 + self._value
                return self._value > SoundFontMath.non_audible()
            case EnvelopeStage.RELEASE:
                self._value = self._release_level * SoundFontMath.exp_cutoff(
                    self._release_slope
                    * (current_time - self._release_start_time)
                )
                self._priority = self._value
                return self._value > SoundFontMath.non_audible()

    @property
    def value(self) -> float:
        return self._value

    @property
    def priority(self) -> float:
        return self._priority


class ModulationEnvelope:
    def __init__(self, synthesizer: "Synthesizer") -> None:
        self._synthesizer = synthesizer

    def start(
        self,
        delay: float,
        attack: float,
        hold: float,
        decay: float,
        sustain: float,
        release: float,
    ) -> None:
        self._attack_slope = 1 / attack
        self._decay_slope = 1 / decay
        self._release_slope = 1 / release

        self._attack_start_time = delay
        self._hold_start_time = self._attack_start_time + attack
        self._decay_start_time = self._hold_start_time + hold

        self._decay_end_time = self._decay_start_time + decay
        self._release_end_time = release

        self._sustain_level = SoundFontMath.clamp(sustain, 0, 1)
        self._release_level = 0

        self._processed_sample_count = 0
        self._stage = EnvelopeStage.DELAY
        self._value = 0

        self.process(0)

    def release(self) -> None:
        self._stage = EnvelopeStage.RELEASE
        self._release_end_time += (
            float(self._processed_sample_count)
            / self._synthesizer.sample_rate
        )
        self._release_level = self._value

    def process(self, sample_count: int) -> bool:
        self._processed_sample_count += sample_count

        current_time = (
            float(self._processed_sample_count)
            / self._synthesizer.sample_rate
        )

        while self._stage <= EnvelopeStage.HOLD:
            end_time: float

            match self._stage:
                case EnvelopeStage.DELAY:
                    end_time = self._attack_start_time
                case EnvelopeStage.ATTACK:
                    end_time = self._hold_start_time
                case EnvelopeStage.HOLD:
                    end_time = self._decay_start_time
                case _:
                    raise ValueError("Invalid envelope stage.")

            if current_time < end_time:
                break
            else:
                self._stage = EnvelopeStage(self._stage.value + 1)

        match self._stage:
            case EnvelopeStage.DELAY:
                self._value = 0
                return True
            case EnvelopeStage.ATTACK:
                self._value = self._attack_slope * (
                    current_time - self._attack_start_time
                )
                return True
            case EnvelopeStage.HOLD:
                self._value = 1
                return True
            case EnvelopeStage.DECAY:
                self._value = max(
                    self._decay_slope
                    * (self._decay_end_time - current_time),
                    self._sustain_level,
                )
                return self._value > SoundFontMath.non_audible()
            case EnvelopeStage.RELEASE:
                self._value = max(
                    self._release_level
                    * self._release_slope
                    * (self._release_end_time - current_time),
                    0,
                )
                return self._value > SoundFontMath.non_audible()

    @property
    def value(self) -> float:
        return self._value
