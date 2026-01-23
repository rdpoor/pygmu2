"""
Example 10: Strudel Sample Map - Lazy downloading and playback

Downloads a Strudel-style JSON map, resolves one sample, and plays
the entire sample. Samples are cached on demand.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pygmu2 import SampleMap, WavReaderPE, AudioRenderer, LoopPE, MixPE, DelayPE

STRUDEL_JSON_URL = "https://software.tomandandy.com/strudel.json"
SAMPLE_NAME = "gettingAction"
LOOP_PATTERN = "loop?"
LOOP_COUNT = 4

print("=== pygmu2 Example 10: Strudel Sample Map ===", flush=True)
print(f"Fetching map: {STRUDEL_JSON_URL}", flush=True)
samples = SampleMap.from_url(STRUDEL_JSON_URL)
sample_path = samples.resolve(SAMPLE_NAME)
loop_path = samples.resolve(LOOP_PATTERN)
print(f"Resolved '{SAMPLE_NAME}': {sample_path}", flush=True)
print(f"Resolved '{LOOP_PATTERN}': {loop_path}", flush=True)

source = WavReaderPE(sample_path)
loop_source = WavReaderPE(loop_path)
looped = LoopPE(loop_source, count=LOOP_COUNT)
mixed = MixPE(
    source, 
    DelayPE(source, delay=loop_source.extent().end), 
    DelayPE(source, delay=2 * loop_source.extent().end), 
    DelayPE(source, delay=3 * loop_source.extent().end), 
    looped)
file_sr = source.file_sample_rate
renderer = AudioRenderer(sample_rate=file_sr)
renderer.set_source(mixed)

extent = mixed.extent()
duration_samples = extent.end - extent.start
duration_seconds = duration_samples / file_sr
print(f"Playing {duration_seconds:.2f} seconds...", flush=True)
with renderer:
    renderer.start()
    renderer.play_extent()

print("Done!", flush=True)
