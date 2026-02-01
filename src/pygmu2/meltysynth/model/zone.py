from typing import Sequence

from pygmu2.meltysynth.model.generator import Generator
from pygmu2.meltysynth.model.zone_info import ZoneInfo


class Zone:
    def __init__(self, generators: Sequence[Generator]) -> None:
        self._generators = generators

    @staticmethod
    def create(
        infos: Sequence[ZoneInfo], generators: Sequence[Generator]
    ) -> Sequence["Zone"]:
        if len(infos) <= 1:
            raise Exception("No valid zone was found.")

        count = len(infos) - 1
        zones: list[Zone] = []

        for i in range(count):
            info = infos[i]
            gs: list[Generator] = []
            for j in range(info.generator_count):
                gs.append(generators[info.generator_index + j])
            zones.append(Zone(gs))

        return zones

    @staticmethod
    def empty() -> "Zone":
        return Zone([])

    @property
    def generators(self) -> Sequence[Generator]:
        return self._generators
