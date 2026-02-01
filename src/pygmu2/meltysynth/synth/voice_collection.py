from collections.abc import MutableSequence
from typing import TYPE_CHECKING

from pygmu2.meltysynth.model.instrument import InstrumentRegion
from pygmu2.meltysynth.synth.voice import Voice

if TYPE_CHECKING:
    from pygmu2.meltysynth.synth.synthesizer import Synthesizer


class VoiceCollection:
    def __init__(
        self, synthesizer: "Synthesizer", max_active_voice_count: int
    ) -> None:
        self._synthesizer = synthesizer
        self._voices: MutableSequence[Voice] = []
        for _ in range(max_active_voice_count):
            self._voices.append(Voice(synthesizer))
        self._active_voice_count = 0

    def request_new(
        self, region: InstrumentRegion, channel: int
    ) -> Voice:
        exclusive_class = region.exclusive_class

        if exclusive_class != 0:
            for i in range(self._active_voice_count):
                voice = self._voices[i]
                if (
                    voice.exclusive_class == exclusive_class
                    and voice.channel == channel
                ):
                    return voice

        if self._active_voice_count < len(self._voices):
            free = self._voices[self._active_voice_count]
            self._active_voice_count += 1
            return free

        candidate = self._voices[0]
        lowest_priority = 1000000.0

        for i in range(self._active_voice_count):
            voice = self._voices[i]
            priority = voice.priority
            if priority < lowest_priority:
                lowest_priority = priority
                candidate = voice
            elif priority == lowest_priority:
                if voice.voice_length > candidate.voice_length:
                    candidate = voice

        return candidate

    def process(self) -> None:
        i = 0
        while True:
            if i == self._active_voice_count:
                return
            if self._voices[i].process():
                i += 1
            else:
                self._active_voice_count -= 1
                tmp = self._voices[i]
                self._voices[i] = self._voices[self._active_voice_count]
                self._voices[self._active_voice_count] = tmp

    def clear(self) -> None:
        self._active_voice_count = 0

    @property
    def active_voice_count(self) -> int:
        return self._active_voice_count
