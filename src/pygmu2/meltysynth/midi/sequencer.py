from collections.abc import MutableSequence
from typing import Optional, TYPE_CHECKING

from pygmu2.meltysynth.math_utils import create_buffer
from pygmu2.meltysynth.midi.message import MidiMessageType
from pygmu2.meltysynth.midi.midi_file import MidiFile

if TYPE_CHECKING:
    from pygmu2.meltysynth.synth.synthesizer import Synthesizer


class MidiFileSequencer:
    def __init__(self, synthesizer: "Synthesizer") -> None:
        self._synthesizer = synthesizer
        self._midi_file: Optional[MidiFile] = None
        self._block_wrote: int = 0
        self._current_time: float = 0.0
        self._msg_index: int = 0
        self._block_left: Optional[MutableSequence[float]] = None
        self._block_right: Optional[MutableSequence[float]] = None

    def play(self, midi_file: MidiFile, loop: bool) -> None:
        self._midi_file = midi_file
        self._loop = loop
        self._block_wrote = self._synthesizer.block_size
        self._current_time = 0.0
        self._msg_index = 0
        self._block_left = create_buffer(self._synthesizer.block_size)
        self._block_right = create_buffer(self._synthesizer.block_size)
        self._synthesizer.reset()

    def stop(self) -> None:
        self._midi_file = None
        self._synthesizer.reset()

    def render(
        self,
        left: MutableSequence[float],
        right: MutableSequence[float],
        offset: Optional[int] = None,
        count: Optional[int] = None,
    ) -> None:
        if len(left) != len(right):
            raise Exception(
                "The output buffers for the left and right must be the same length."
            )
        if offset is None:
            offset = 0
        elif count is None:
            raise Exception("'count' must be set if 'offset' is set.")
        if count is None:
            count = len(left)

        wrote = 0
        while wrote < count:
            if self._block_wrote == self._synthesizer.block_size:
                self._process_events()
                self._block_wrote = 0
                self._current_time += (
                    self._synthesizer.block_size
                    / self._synthesizer.sample_rate
                )

            src_rem = self._synthesizer.block_size - self._block_wrote
            dst_rem = count - wrote
            rem = min(src_rem, dst_rem)

            self._synthesizer.render(left, right, offset + wrote, rem)

            self._block_wrote += rem
            wrote += rem

    def _process_events(self) -> None:
        if self._midi_file is None:
            return
        while self._msg_index < len(self._midi_file._messages):
            time = self._midi_file._times[self._msg_index]
            msg = self._midi_file._messages[self._msg_index]

            if time <= self._current_time:
                if msg.type == MidiMessageType.NORMAL:
                    self._synthesizer.process_midi_message(
                        msg.channel, msg.command, msg.data1, msg.data2
                    )
                self._msg_index += 1
            else:
                break

        if (
            self._msg_index == len(self._midi_file._messages)
            and self._loop
        ):
            self._current_time = 0.0
            self._msg_index = 0
            self._synthesizer.note_off_all(False)
