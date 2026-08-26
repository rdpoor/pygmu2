# pygmu2 — Development Plan

**From:** System Architect
**Governed by:** `DESIGN_PHILOSOPHY.md` (senior document; its §5 verdicts are applied here, not
re-argued). Re-derived 2026-08-26 at commit `c5090e9`; supersedes the previous version of this
file, whose DD-2/DD-5 decisions and Phase-2 placement of the consumer gates were overridden.

**The shape of this plan in one sentence:** consumer gates first, then a phase that is almost
entirely deletions, then the contract suite that carries correctness from that point on, then
surface/consumer repair, profiling, and docs — with debugging aids and optimizations explicitly
*not* planned (they are second-pass purchases per philosophy §4).

**Owner directive (2026-08-26):** backward compatibility is not a requirement. Where a module
is clearer rewritten than patched, rewrite it — the tests and consumer gates, not diff
minimality, are what protect correctness. Concretely: after the Phase-1 deletions, `config.py`
(~20 surviving lines) and `renderer.py` (lifecycle walkers + abstract `_output`) are rewritten
clean rather than left with scar tissue; `AudioRenderer` (P4.3) is rebuilt around a single
stream model rather than merging two paths; tests of deleted machinery are rewritten, not
patched.

---

## 0. Baseline (measured at `c5090e9`)

| Signal | Value | Target |
|---|---|---|
| Tests | 1433 passing, 8 skipped, 69% cov | green throughout; contract coverage for every PE |
| `import pygmu2` | 740 ms (scipy 426, numba 146) | <100 ms, no scipy/numba |
| `black --check` | 158/198 files fail | 0 |
| CI / pre-commit | none | CI gates per philosophy §8 |
| `python -m pygmu2` | broken (`hello`) | fixed or deleted |
| `benchmarks/benchmark_pes.py` | broken (`RandomPE`) | runs; smoke-tested in CI |
| Examples | 8 `.py-disabled`, 7 dead README rows, 31 undocumented | 0 disabled; tables generated |
| `processing_element.py` + `renderer.py` | 343 + 583 lines | **smaller than baseline** |
| Prose-only invariants | `is_pure`, "framework enforces", `required_input_channels`, immutability | zero, or gated |

---

## 1. Architect decisions (AD)

The philosophy leaves two mechanisms to the Architect. Recorded here so the code review can
cite them.

### AD-1 — Statefulness is a one-line class attribute, falsified by CI

The contiguity check (philosophy §3 R2) must know which PEs hold render state; a stateless PE
is legitimately random-access and must not be constrained. Deleting `is_pure()` removes the old
(two-meanings, unfalsifiable) declaration; the check still needs *one* bit.

Decision: a class attribute on `ProcessingElement`:

```python
class ProcessingElement(ABC):
    stateful: bool = False   # True ⇒ render() calls must be contiguous
```

Stateful PEs set `stateful = True` — one line, replacing their entire `is_pure()` method. This
is nominally a declaration, which R1 discourages; it is admissible because R1's actual rule is
that any declaration must be *falsifiable by a test*, and Tier 1's randomised-order test is
exactly that: a PE claiming `stateful = False` must reproduce the contiguous reference under
shuffled access, and a PE claiming `stateful = True` must raise. A lie in either direction
fails CI. The conversion is mechanical: `is_pure() → False` becomes `stateful = True` (27
sites); `is_pure() → True` becomes deletion (14 sites); the 5 delegating overrides become
whichever their internal graph warrants.

Default is `False`: a forgotten declaration produces wrong audio under seeks — which is
precisely what the Tier 1 test detects, so CI converts the mistake from silent to loud.

### AD-2 — `CachePE` declares `stateful = False`; the contiguity check makes that safe

`CachePE` holds a cache, but it is order-safe *by design*: identical repeated requests serve
from cache, and a diverging request re-renders the source — where, if the source is stateful,
the source's own contiguity check raises. The failure therefore surfaces at the PE that owns
the state, with `repr()` of the actual offender. No special-casing in the base class, no
repeat-request carve-out in the check itself. `CachePE.is_pure()`'s apologetic comment block is
deleted along with the method.

### AD-3 — Contiguity check semantics (the exact one-liner's contract)

In `ProcessingElement.render()`, before dispatch to `_render()`:

- if `self.stateful` and a previous render occurred and `start != self._expected_start`:
  `raise RuntimeError(f"{self!r}: non-contiguous render: expected start "
  f"{self._expected_start}, got {start}")`
- `_expected_start` updates to `start + duration` after each render; it clears in `on_start()`,
  `on_stop()`, and `reset_state()` — so an explicit `reset_state()` is the sanctioned seek.
- `duration == 0` requests bypass the check (they render nothing), matching the existing
  zero-duration early-out.

Error text carries `repr(self)` only — no names machinery (philosophy §4 item 1).

---

## 2. Phases

Ordering follows philosophy §5.3: consumer gates precede the contract work, because Phase 3's
blast radius is what the gates are for.

---

### Phase 0 — Consumer gates and unblocking

*Everything here is independent; land as separate small commits. Nothing depends on later
phases.*

**P0.1 — `black` repo-wide, one isolated commit.** Exclude `src/pygmu2/meltysynth/` (vendored)
via `[tool.black] extend-exclude`. Clears ~70% of lint debt mechanically; do it first, on the
currently-clean tree, so no later diff is polluted.

**P0.2 — Fix the benchmark suite.** `benchmarks/benchmark_pes.py:233` imports `RandomPE`,
`TriggerPE`, `AdsrPE` — propagate the renames (`RandomValuePE`, `PeriodicTrigger`,
`AdsrGatedPE`, …). Add configs for PEs added since. This also produces the per-PE constructor
registry that Tier 1 will reuse (P2.1).

**P0.3 — Examples smoke test** (`tests/test_examples.py`): exec every `examples/*.py` with
`__name__ != "__main__"`; assert no exception. The 8 `.py-disabled` files are *excluded* here
and become P3.4's inventory. Headless — no audio device is touched at import time (verify;
`examples_helper` only plays inside demo functions).

**P0.4 — Benchmark smoke test** (`tests/test_boundaries.py`): run
`benchmark_pes.py --list` in a subprocess; assert exit 0. Same file later hosts the R4/R7
checks (P2.3–P2.5).

**P0.5 — CI** (`.github/workflows/ci.yml`): on push/PR — `uv sync`, `uv run pytest -q`
(includes the new smoke tests), `uv run black --check src tests examples benchmarks scripts`,
and a step failing on any `*.py-disabled` file newer than this commit (`git diff --diff-filter=A`).
This is the acceptance-test gate: after P0, an un-propagated rename cannot land green.

**P0.6 — Unblock trivia.** Fix `src/pygmu2/__main__.py` (delete the `hello` import; print
version + device list, or delete the module). Remove `--cov --cov-report=html` from
`[tool.pytest.ini_options] addopts` in `pyproject.toml`; coverage moves to an explicit CI step.

*Exit criteria:* CI green; pushing a branch that renames a public symbol without propagating it
**fails CI**. That is the philosophy's acceptance test, available from Phase 0 onward.

---

### Phase 1 — Deletions and the two checks

*The contracts phase. Per philosophy §5.2, every item is a deletion or a one-liner. Land in the
order below; each step keeps the suite green.*

**P1.1 — Delete `ErrorMode`.** In `src/pygmu2/config.py`: remove `ErrorMode`,
`DEFAULT_ERROR_MODE`, `set_error_mode`, `get_error_mode`, `handle_error`; keep
`set_sample_rate`/`get_sample_rate`. Convert the 33 `handle_error()` call sites to plain
`raise` (or, where the call guarded an idempotent no-op like `Renderer.stop()`, keep the plain
early-return). Remove the four names from `__init__.py`/`__all__`. Update
`tests/test_config.py` and any LENIENT-mode tests. README's Error Handling section dies in
P5.2.

**P1.2 — Delete `_validate_graph` and the channel-requirement machinery.** In
`src/pygmu2/renderer.py`: `set_source()` becomes assignment + logging — no walk. Delete
`_validate_graph`, and delete the now-dead `self._channel_count` plumbing. In
`processing_element.py`: delete `required_input_channels()` and `resolve_channel_count()`.
Channel mismatches now surface as numpy's own broadcast error at render time — accepted by
decision (philosophy §5.2).

**P1.3 — Delete the `Renderer` profiler.** `PEProfile`, `ProfileReport`, `enable_profiling`,
`disable_profiling`, `get_profile_report`, `print_profile_report`, `_render_profiled`,
`_pe_list`, and `_collect_pes` if nothing else uses it (only profiling did). `renderer.py`
shrinks by ~200 lines. Check `scripts/profile_score.py` and benchmarks for callers first;
retarget them to `diagnostics.py` (P4.1).

**P1.4 — Lazy channel width in all playback paths.** `AudioRenderer._output()` already opens
its stream from the first snippet's width (`audio_renderer.py:103-112`). Port the same pattern
to `play_extent()` and `stream_start()`: render (or receive) the first block, open the stream
with `snippet.channels`, then continue. The `channel_count` property on `Renderer` is deleted
with P1.2; `channel_count()`/`resolve_channel_count()` on PEs remain only where PEs themselves
consume them — audit and prune.

**P1.5 — The two R2 checks in the base class** (`processing_element.py`):
- AD-3's contiguity check (~8 lines including the raise).
- `Snippet.__init__`: `data.flags.writeable = False` after dtype normalisation (one line).
  Expect fallout where a PE constructs a Snippet and then mutates the same array it passed in,
  or mutates an input's buffer — each is a bug this check exists to expose; fix at the site.

**P1.6 — The `is_pure` → `stateful` sweep** (AD-1). One commit, mechanical: delete `is_pure()`
from the base class and its 42 catalog implementations + 5 delegations; add `stateful = True`
to the 27 stateful PEs. Correct the false comment at `sine_pe.py:204`. Delete the three
hand-rolled contiguity guards now subsumed by the base class: `spatial_pe.py:768`
(`_reset_tail_if_noncontiguous`), `convolve_pe.py:256`, `analog_osc_pe.py:160` — each becomes
"trust the base class; state resets only via `reset_state()`".

**P1.7 — Simplify `CachePE`** (AD-2): delete `is_pure()` and its comment; keep the single-entry
memoization and lifecycle hooks unchanged.

**P1.8 — Fallout triage.** The suite's 138 non-zero-start `render()` calls plus any examples
that seek: each either (a) legitimately seeks → insert `reset_state()` or a fresh instance, or
(b) exposed a latent bug → fix. Budget the largest share of Phase 1 time here; the mechanical
edits above are hours, the triage is the real work.

*Exit criteria:* suite green; `grep -rE "is_pure|_validate_graph|required_input_channels|ErrorMode|handle_error" src/pygmu2 --exclude-dir=meltysynth`
returns nothing; `wc -l` of `processing_element.py` + `renderer.py` is below 343 + 583.

---

### Phase 2 — The contract suite (Tier 1) and remaining CI gates

*After Phase 1, runtime validation is gone; this phase builds the thing that replaces it.*

**P2.1 — PE factory registry.** Tier 1 needs to construct every PE with valid arguments. Reuse
the config pattern (and configs) from `benchmarks/benchmark_pes.py` (P0.2) rather than building
a second registry — R4. Home: `tests/conftest.py` or `tests/pe_factories.py`, imported by both
the contract suite and the benchmarks so they cannot drift apart.

**P2.2 — `tests/test_contract.py`**, parametrised over every PE discovered from `__all__`
(plus the lazy-import names). Cases per philosophy §6 Tier 1:
- exact `(start, duration)` framing; 2-D float32; expected channel count; **buffer read-only**
- zero-fill outside `extent()`; `extent()` stable across lifetime
- `on_start()`/`on_stop()` idempotent; `reset_state()` restores first-render behaviour
- **randomised-order**: shuffled `(start, duration)` pulls either match a contiguous reference
  (PE behaves stateless) or raise the AD-3 error (PE declared `stateful = True`) — assert the
  declaration matches the behaviour in both directions. Fixed seed; no wall-clock dependence.

**P2.3 — Export-completeness test** (R7): walk `src/pygmu2/*.py`; every public `*PE` class is
in `__all__`/lazy registry or on a commented opt-out list. Forces the P3.2 decisions.

**P2.4 — Generated README tables** (R4): a small script (`scripts/gen_readme_tables.py`)
renders the PE table from `__all__` + first docstring lines, and the examples table from
`examples/`; a Tier 3 test asserts README's generated blocks match the script's output.

**P2.5 — Import-hygiene test**: `import pygmu2` in a subprocess; assert `scipy` and `numba`
absent from `sys.modules` and cold import under the budget (generous bound, e.g. 300 ms, to
avoid CI flake; the <100 ms target is verified manually in §4). Lands with P3.1 in the same PR,
since it fails until the registry is complete.

**P2.6 — PR template** (`.github/pull_request_template.md`) with the two scripted questions
from philosophy §8: "what test fails if this claim is violated?" and "which measured error or
profile motivates this aid/optimization?"

*Exit criteria:* every exported PE parametrised in `test_contract.py` with zero skips; a PE
whose `stateful` declaration lies fails CI in either direction.

---

### Phase 3 — Surface and consumers

*Parallelisable across the two maintainers; each item is an independent PR.*

**P3.1 — Complete the lazy-import registry** (`__init__.py`): add `BlitSawPE`, `EnvelopePE`,
`LadderPE`/`LadderMode`, `CombPE`, `WindowPE`/`WindowMode`, the `SpatialPE` family,
`ReversePitchEchoPE`, `MeltysynthPE`, `MidiInPE`, `NotesPE`. Chase transitive leaks (a lazy PE
imported by an eager module isn't lazy). Gate: P2.5.

**P3.2 — Close export holes.** Add `DecayingSinePE`, `IdiophonePE` to `__all__`. Decide, per
philosophy R7 (export or delete, no limbo): `portamento_pe`, `extent_window_pe`, `annotations`
(sole `pyyaml` consumer), `interpolated_lookup`, `nofft_spectrogram` (used by
`scripts/jogshuttle.py` — likely export). Gate: P2.3.

**P3.3 — One dependency manifest.** Delete `requirements.txt` (empty stub) and
`Pipfile`/`Pipfile.lock` (drift; bogus `rtmidi` entry). Move `numba`, `mido`, `python-rtmidi`,
`miniaudio` to extras consistent with P3.1's laziness; fix the duplicate `miniaudio`
declaration; reconcile `version` (0.1.0 vs 0.2.0 — pick 0.2.0, single source in
`pyproject.toml`, `__init__.py` reads it via `importlib.metadata`); align the Python floor
(3.10 vs 3.12 across four configs). Import-time errors for missing extras: whatever
`ImportError` naturally occurs, per PD-2 — no wrapper messages in the first pass.

**P3.4 — Resolve all 9 `.py-disabled` files** (8 examples + `scripts/toy_midi_sampler`): five
need rename propagation (`13_random`, `14_trigger`, `18_adsr`, `25_gating`, `31_trigger`);
three import cleanly and are restore-or-delete decisions vs. their successors
(`19_sequence_examples.py`, `37_sequence_eg.py`). Zero `.py-disabled` files remain; P0.5's CI
step prevents new ones.

**P3.5 — One example naming convention:** `topic_eg.py` (the convention of every recent
commit; removes the `19_`/`20_`/`21_` collisions by construction). One rename commit after
P3.4, then regenerate the README table (P2.4 script).

**P3.6 — `examples_helper.run_demos()`:** fix the `demos: dict` annotation to
`list[tuple[str, Callable]]`; route the `argv` single-demo path through `run_one_demo()` so the
banner prints (closes the BACKLOG item); then strip redundant leading `print()`s from demo
bodies.

**P3.7 — `utils.browse()`:** currently hard-codes `uv run` and the source-tree path
(`utils.py:147-165`). Simplest conforming fix: raise a clear error when `jogshuttle.py` isn't
found, rather than building install-aware discovery — a runtime error is acceptable (PD-2);
packaging the viewer is second-pass.

---

### Phase 4 — Profiling (the PD-2 sanctioned mechanism)

**P4.1 — Promote `diagnostics.py`.** Export from `__init__.py`; add a context-manager façade
(`with pg.diagnostics.profile() as report:`); write its first tests; adopt the
realtime-ratio/samples-per-second presentation salvaged from the deleted `ProfileReport`.
Retarget `scripts/profile_score.py` and the benchmark suite's timing to it.

**P4.2 — Render-mode switching** (BACKLOG): `PYGMU_RENDER_MODE={realtime|offline|profile}`
read by `examples_helper`, swapping `pg.play` → `pg.play_offline` → a diagnostics-wrapped
offline render with a report printed at exit. No example-script changes.

**P4.3 — One playback path in `AudioRenderer`** (R4): `play_extent()` currently opens its own
stream ignoring `blocksize`/`latency`; consolidate on the long-lived `_output()` stream. While
there, verify `stream_start`'s callback end-of-extent arithmetic (`frames` reassignment before
`_stream_position += frames`). Exceptions in the callback surface however they surface —
graceful capture is second-pass item 4.

**P4.4 — Tier 4 seeds.** Grow `tests/test_analytical_pe.py` toward the philosophy §6 Tier 4
list (oscillator THD, filter −3 dB points, convolution IR length, dynamics gain accuracy) —
opportunistically, one property per touched PE family; not a gating deliverable.

---

### Phase 5 — Documentation and remaining tooling

**P5.1 — `CONTRIBUTING.md` rewrite.** Delete the Purity section; document the `stateful`
attribute, the contiguity contract, and `reset_state()` as the sanctioned seek. Rewrite the
composite-PE pattern around AD-2 (CachePE as fan-out memoizer; errors surface at the stateful
source). New-PE checklist at **≤5 steps**: (1) write `<name>_pe.py`, set `stateful` if it holds
state; (2) add to `__init__.py`/`__all__`; (3) add a factory entry (P2.1) — Tier 1 coverage
and benchmarks follow automatically; (4) add behavioural tests for its DSP; (5) run
`uv run pytest`. Fix the stale project tree (`ramp_pe.py`, `random_pe.py`).

**P5.2 — `README.md`:** swap in the generated tables; delete the Error Handling section;
verify Quick Start against the surviving API.

**P5.3 — Lint/type ratchet:** flake8 config (`max-line-length = 88`, exclude vendored
`meltysynth/`), fix the 79 F401 unused imports; mypy strict on the Phase-1 core files only
(`processing_element`, `snippet`, `extent`, `renderer`, `config`), growing the allowlist
opportunistically. Pre-commit: `black` + `flake8` on changed files.

**P5.4 — Groom:** fold done items out of `BACKLOG.md`; move `TEMPERAMENT_IMPLEMENTATION.md`
to `docs/`; `DESIGN_PHILOSOPHY.md` stays at root as the governing reference.

---

## 3. Explicitly not planned (second-pass ledger)

Per philosophy §4 these are *not* scheduled; each requires attached evidence (a measured
hard-to-debug error, or a profile) to enter a future milestone:

1. User-assignable `name=` on PEs — first pass uses `repr(self)`
2. Shape checks with PE names in `MixPE` et al. — first pass keeps numpy's broadcast error
3. Cycle guard with path trace — first pass keeps `RecursionError`
4. Graceful callback-thread error capture in `AudioRenderer`
5. `PatchCable`
6. Any numba/scipy acceleration of new code
7. Configurable error handling (post-`ErrorMode`)

Feature work from BACKLOG (`SpatialPE` TODO stubs, `TralfamPE` zero-padding/normalisation,
HRTF demo polish) is orthogonal to the overhaul and proceeds under the change protocol
(philosophy §7) whenever scheduled.

---

## 4. Verification

Philosophy §10 is the acceptance checklist; run it verbatim at the end of each phase. Phase
mapping of its rows:

| Philosophy §10 row | Satisfied by |
|---|---|
| CI fails un-propagated rename | **P0.5** (available before the risky work starts) |
| Consumers green | P0.3, P0.4 |
| Disabled files 9 → 0 | P3.4 |
| Deleted machinery grep = 0 | P1.1–P1.7 |
| Contract coverage, zero skips | P2.2 |
| Untested modules 28 → 0 (contract level) | P2.2 |
| Dangling old names = 0 | P0.2 + P3.4 + P5.2 |
| Import < 100 ms, no scipy/numba | P3.1 (gated by P2.5) |
| Checklist ≤ 5 steps | P5.1 |
| Core `wc -l` below 343 + 583 | Phase 1 exit criteria |

## 5. Risks

| Risk | Handling |
|---|---|
| P1.8 triage larger than expected (138 suite call sites + unknown example seeks) | P0's gates are in place first, so breakage is loud and enumerable; triage list is generated mechanically by running the suite after P1.5 |
| `writeable=False` fallout in PEs mutating post-construction | Same: run suite after P1.5, fix at each raise site; the raise *is* the diagnostic |
| `stateful` mis-declared during the P1.6 sweep | Undetected only until P2.2 lands; sequence P2.2 immediately after Phase 1, before any Phase 3 parallel work builds on the catalog |
| P3.3 extras break a working environment | Pre-1.0, accepted; natural `ImportError` per PD-2 |
| Deleting `Renderer` profiler breaks a caller | P1.3 greps `scripts/` and `benchmarks/` first; P4.1 restores capability via `diagnostics.py` in the same milestone |
| Vendored `meltysynth/` swept accidentally | Excluded by name in black/flake8/grep targets and the P1.6 sweep |
