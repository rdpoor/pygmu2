# pygmu2 Backlog

Items are listed roughly in priority order within each section.

## Features

- **TralfamPE**: Add zero padding and automatic gain normalization.

- **Render mode switching**: Design and implement a mechanism to switch between real-time rendering, offline rendering, and profiled rendering without modifying user scripts. A `--render-mode` CLI flag (or environment variable) read by `examples_helper` could swap out `pg.play` at startup. Profiled mode could run `NullRenderer` under `cProfile`, or — more usefully — instrument each PE's `render()` call to accumulate wall-clock time, then print a per-PE time summary at the end.

## Cleanup / Refactoring

- **`27_spatial.py`**: Review and improve `demo_hrtf_spatialization`.

- **Examples**: For all examples using `examples_helper.run_demos()`, print the demo name string before running each demo. If a demo function already starts with a `print()`, consider using that text as the name string (and remove the redundant print from the function body).

## Examples

## Tests

## Documentation
