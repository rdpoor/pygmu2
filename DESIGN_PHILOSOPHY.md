# pygmu2 Overhaul — Design Rules, Philosophy & Testing Methodology

**From:** Technical Manager
**To:** System Architect
**Status:** Senior document. Where this conflicts with `IMPLEMENTATION_PLAN.md`, this governs — see §5 for the specific verdicts you must act on.

---

## 1. Context

pygmu2 is a lazy-evaluation DSP library: a DAG of `ProcessingElement`s pulled on demand via
`render(start, duration) -> Snippet`. It has ~122 modules, 1433 green tests, and two
maintainers. It is pre-1.0 and `Development Status :: Alpha`.

An architectural audit at commit `c5090e9` produced `IMPLEMENTATION_PLAN.md`. This document
exists because that plan identified *what* to fix without establishing *why those fixes and not
others* — and without a rule that prevents the same debt reaccumulating. The Architect is asked
to overhaul the codebase; this establishes the constraints that overhaul must satisfy.

**Operating parameters set by the project owner:**

| Parameter | Setting | Consequence for this document |
|---|---|---|
| Authority | This document constrains; re-derive the plan against it | §5 records confirmations, overrides, and resolutions |
| API stability | Free rein pre-1.0 — no deprecation shims | Deletion is the preferred fix. Renaming is cheap; carrying two vocabularies is not |
| Enforcement | Machine gates, not review discipline | **A rule that isn't a gate is a suggestion.** Every rule below names its gate |
| Product | Reusable DSP library; the PE catalog *is* the product | Examples are documentation. Public API surface is product surface. Contract testing is the centerpiece |
| Simplicity | Core design as simple as possible; runtime errors are acceptable; no pre-optimization; no up-front validation machinery | The second prime directive (§2). Debugging aids and optimizations are second-pass purchases (§4) |
| Error handling | `ErrorMode` (STRICT/LENIENT) is **deleted** — always raise | Removes `config.handle_error()` machinery; formalises the existing 109-direct-`raise` : 33-`handle_error` reality |

### The diagnosis, in one sentence

**pygmu2's debt is not bad code — it is invariants written in prose that nothing reads.**

The evidence is consistent. `src/` contains exactly two `TODO` markers outside vendored code and
passes 1433 tests. But: `is_pure()` claims a guarantee the framework never enforces;
`processing_element.py:180` states "the framework enforces this" about contiguous rendering that
nothing checks; `CONTRIBUTING.md` declares Snippet buffers immutable with no `writeable=False`;
`required_input_channels()` is declared once, returns `None`, and its validation loop can never
fire; and a library-wide rename propagated to nothing downstream because nothing downstream was
under test. Every one of these is prose the machine doesn't read.

Frame the overhaul accordingly. This is not a cleanliness campaign — it is a campaign to make
claims executable and boundaries verified, **using the simplest mechanisms that can fail loudly.**

---

## 2. Prime directives

Two directives. The first says what a claim must be; the second says where each kind of work
belongs.

> **PD-1 — Executable or deleted.**
> Every invariant this codebase claims is either enforced by a machine or it does not exist.
> There is no third category of "documented convention."

When you find a prose-only invariant, you have exactly two moves: give it a gate, or delete the
claim. Documenting it harder is not a move. Pre-1.0 freedom (§1) means deletion is nearly always
available, and is usually correct — an unenforced guarantee is worse than no guarantee, because
downstream code is written trusting it. `sine_pe.py:204` is the canonical case: a PE that skips
its own safety check because it believes a promise the base class never kept.

> **PD-2 — Validation lives in CI; errors live at render time; optimization waits for a profile.**
> The render path is the product. Keep it bare: no pre-validation walks, no defensive
> scaffolding, no speculative fast paths. Correctness is proven exhaustively in the test suite;
> at runtime, things simply fail — loudly, where the fact surfaces.

PD-2 is the simplicity amendment, and it resolves what would otherwise be a tension inside PD-1:
gates are mandatory, but they live in CI, not in the signal path. The two directives compose
into a division of labour:

| Concern | Where it lives | What runtime does |
|---|---|---|
| Correctness | Tier 1 contract suite, in CI, over every PE | nothing — trusts CI |
| Structural failure (bad graph, channel mismatch, cycles) | surfaces naturally at render | raises whatever error occurs, unwrapped |
| Silent corruption (wrong audio with no error) | the **one** exception: two one-line runtime checks (§3 R2) | contiguity compare; read-only buffers |
| Diagnosis | second pass, purchased by evidence (§4) | `repr(self)` in error text — already free |
| Performance | `diagnostics.py` profiling, on demand | zero-cost hooks when disabled |

---

## 3. Design rules

Six rules. Each states the rule, the evidence that motivates it, and the gate that enforces it.
Rules are numbered for citation in review.

### R1 — Check facts, not declarations

A gate that reads a self-reported flag tests the author's belief, not reality. Prefer checks on
observed behaviour.

*Evidence:* `GainPE.is_pure()` returns `True` because its arithmetic is stateless; a `GainPE`
over a phase-accumulating source returns different data for identical `(start, duration)`
arguments. The declaration is sincere and wrong. No amount of care fixes a category of check
whose input is an opinion.

*Gate:* the universal contract suite (§6, Tier 1) tests behaviour, not declarations. Any new
`is_*()` predicate on `ProcessingElement` requires a test that could falsify it.

### R2 — Fail where the fact surfaces; never fail silently

Do not build machinery to fail earlier than the fact naturally appears. An error raised at
render time, by the code that hit it, is the correct outcome — not a fallback to apologise for.
Include the context already at hand (`repr(self)` exists and costs nothing); build nothing more
until the second pass (§4).

*Evidence for accepting late errors:* the alternative — `_validate_graph` — walks the graph at
`set_source()` reading declarations (R1 violation), and its channel-checking half has been dead
code since inception: `required_input_channels()` is declared once and returns `None`.

*Evidence for the "never silently" clause:* a non-contiguous pull on a phase-accumulating
`SinePE` today returns **wrong audio with no error** (measured: max abs error 0.057 against
ground truth). Silent corruption is the one failure mode PD-2 does not license — it never
becomes an error, so nothing can trigger a second-pass fix for it.

*Therefore the render chokepoint keeps exactly two checks, both one-liners, both runtime:*
1. the contiguity compare — a stateful PE pulled out of order raises instead of returning
   wrong samples;
2. `data.flags.writeable = False` on `Snippet` buffers — in-place mutation of an input raises
   at the write instead of corrupting a sibling sink.

Everything else — channel mismatches, bad graphs, cycles (a `RecursionError` is already loud) —
fails with whatever error naturally occurs, unwrapped, in the first pass.

*Gate:* Tier 1's randomised-order test asserts the contiguity error is raised, not silenced;
a Tier 1 case asserts input buffers are read-only.

### R3 — Never decide silently on the author's behalf

Where the framework could plausibly guess, it raises instead.

*Evidence:* a mono source auto-upmixed to stereo changes the music and never surfaces. So does
a silent state reset, and so did `ErrorMode.LENIENT` — warn, substitute a value, continue.

*Resolution (owner decision):* **`ErrorMode` is deleted.** `handle_error()`, `set_error_mode()`,
and the STRICT/LENIENT enum go; call sites become plain `raise`. This removes machinery (PD-2),
removes a silent-continue mode (R3), and formalises reality — `src/` already has 109 direct
`raise` statements against 33 `handle_error()` calls, so LENIENT was largely non-functional.
README's Error Handling section is deleted with it. Configurable error handling may return as a
second-pass purchase if a concrete need appears.

*Gate:* grep — zero references to `ErrorMode`, `handle_error`, `set_error_mode` in `src/`;
channel-mismatch behaviour covered by Tier 1 (raises, never adapts).

### R4 — One concept, one home

Every capability has exactly one implementation. Two implementations means one is stale and you
won't know which.

*Evidence, all currently live:* two profiling systems (one of which measures nothing it reports);
three dependency manifests; two example naming conventions with three colliding numeric prefixes;
two playback paths in `AudioRenderer` with different `blocksize`/`latency` behaviour; `miniaudio`
declared both as a hard dependency and as an extra, at two different minimum versions. **And one
more this amendment exposed:** the graph walk computes at `set_source()` a channel width that
`AudioRenderer._output()` already discovers for free from the first snippet
(`audio_renderer.py:103-112`) — the walk duplicates the lazy path.

*Gate:* CI checks for the mechanisable cases — one manifest, one naming convention, README
tables generated from `__all__` rather than hand-written.

### R5 — The consumer is part of the system

`src/` is not the system. Examples, benchmarks, and documentation are consumers, and an
unverified consumer is a broken consumer waiting to be discovered.

*Evidence:* the rename wave. It cost five examples, the entire benchmark suite, three test files,
and both README and CONTRIBUTING — and it was *free to make* because nothing downstream ran in
CI. This is the root cause of most of the plan's findings.

*Under a library product model* (§1), examples are documentation, and documentation that doesn't
execute is a claim without a gate — see PD-1.

*Gate:* CI imports every example and runs the benchmark suite. Non-negotiable; this is the
single highest-value gate in the document.

### R6 — Mechanism in the base class, never in the template

Anything the "how to write a new PE" checklist requires, ~110 existing PEs must also do, and
every future PE can forget. Push it into `ProcessingElement` where it is unforgettable — and
under PD-2, keep what's pushed there minimal (today: the two R2 checks and the zero-cost
diagnostics hooks; nothing else).

*Evidence:* PEs override `_render()`, not `render()` — so the base class already has a
chokepoint through which every pull passes, currently used only for a `duration >= 0` check and
diagnostics hooks. That chokepoint is where the two permitted guarantees belong. Three PEs
(`spatial_pe.py:768`, `convolve_pe.py:256`, `analog_osc_pe.py:160`) currently hand-roll private
contiguity guards — the base-class check deletes all three.

*Test of a proposed mechanism:* if it requires editing N PE files, its cost is N and rising. If
it lives in the base class, its cost is 1 and fixed. Prefer the latter decisively; this is the
economics of "easy to extend."

*Gate:* the new-PE checklist in `CONTRIBUTING.md` has a hard budget — **no more than 5 steps.**
If a change would add a sixth, it belongs in the base class instead.

### R7 — The public surface is the product

Under a library model, `__all__`, naming, signatures, and docstrings are not housekeeping.

*Evidence:* `PortamentoPE` and `ExtentWindowPE` are complete, working, untested, and
unreachable — never imported by `__init__.py`. `DecayingSinePE` and `IdiophonePE` are imported
but absent from `__all__` while being used by shipped examples. Nobody noticed, because nothing
checks.

*Gate:* a test walking `src/pygmu2/*.py` asserting every public `*PE` class is either exported
or on an explicit, commented opt-out list. Pre-1.0 (§1), removal is a valid resolution — an
unreachable PE should be deleted or exported, not left in limbo.

---

## 4. The second pass — complexity is purchased, not presumed

PD-2 defers two whole categories of work, on the same trigger discipline:

**Debugging aids** are added only when a runtime error has actually proven hard to debug — not
speculatively. Deferred to this list, in likely order of purchase:

1. User-assignable `name=` on `ProcessingElement` (first pass: errors carry `repr(self)`, which
   already exists and costs nothing)
2. Local shape checks in multi-input PEs (`MixPE` et al.) naming both offending PEs — replacing
   numpy's raw `non-broadcastable output operand` message *if* it proves confusing in practice
3. An in-flight cycle guard with a path trace — replacing the (already loud) `RecursionError`
4. Graceful error capture in `AudioRenderer`'s callback thread (log-and-stop instead of a dead
   stream) — with the constraint that a swallowed exception is a *silent* failure and thus
   forbidden by R2: the minimal first-pass behaviour is that the exception surfaces, however
   ungracefully
5. `PatchCable` — reified, named graph edges. The far end of this axis; `CachePE` is the seed

**Optimizations** are added only when the profiler names a critical section. The sanctioned
mechanism is `diagnostics.py` — per-PE render timing and pull counts via zero-cost-when-disabled
hooks already present in the chokepoint. This elevates the profiling workstream (plan items
B3/B4): delete the `Renderer` profiler that attributes all time to the root PE, export and test
`diagnostics.py`, give it a context-manager façade. **New PE code is plain numpy until measured**;
numba/scipy accelerations in new code require a profile showing the need.

*Gate for both:* the PR template asks one scripted question — "which measured error or profile
motivates this aid/optimization?" A second-pass purchase without evidence attached is rejected.

---

## 5. Re-derivation verdicts on `IMPLEMENTATION_PLAN.md`

Per §1, the plan is not settled. These are the verdicts, including the simplicity amendment's
overrides of the plan's own design decisions (DD-1…DD-5).

### 5.1 Confirmed

| Decision | Derives from | Note |
|---|---|---|
| **DD-1** delete `is_pure()`; contiguity checked at render | R1, R2 | The archetype of the philosophy: a one-line runtime check on a fact, replacing a declaration |
| **DD-3** raise on channel mismatch, never auto-adapt | R3 | Confirmed, and simplified: the raise is numpy's own broadcast error in the first pass — no walk, no wrapper (§4 item 2 is the second-pass upgrade) |
| **DD-4** defer `PatchCable` | R6, PD-2 | Confirmed and extended: the node-side aids (names, cycle guard) that DD-4 traded up to are themselves now second-pass (§4) |

### 5.2 Overridden by the simplicity amendment

| Decision | Verdict | Rationale |
|---|---|---|
| **DD-2** keep the channel walk as `_resolve_channels` | **Overridden — delete `_validate_graph` entirely, walk included** | The walk's validation half reads declarations (R1) and its channel-check half is dead code (§3 R2 evidence). Its one real service — output width — is already discovered lazily by `AudioRenderer._output()` from the first snippet (R4 evidence). `play_extent()` and `stream_start()` adopt the same lazy pattern. Delete `required_input_channels()` and `resolve_channel_count()` with it |
| **DD-5** start-time dry run | **Overridden — cut** | Up-front checking machinery, exactly what PD-2 forbids. A graph CI never exercised can fail mid-render; that is accepted by decision, not oversight |
| Plan §4.3 tension (early-unreliable vs late-reliable check) | **Resolved by owner decision** | The "early" leg moves out of runtime entirely, into CI (Tier 1). No scattered-start dry run, no advisory static warning. Late failure is the design, not a compromise |
| Plan A3 (names) / A4 (cycle guard) in Phase 1 | **Moved to second pass** (§4 items 1, 3) | Debugging aids await evidence. First-pass errors carry `repr(self)` |
| Plan D3 (populate `required_input_channels`) | **Deleted outright** | The mechanism goes, not just the dead declarations |
| Plan D7 (normalise toward `handle_error()`) | **Inverted** | `ErrorMode` is deleted (§3 R3); the sweep converts the 33 `handle_error()` sites to plain `raise`, not the 109 raises to `handle_error()` |

Net effect: **Phase 1 of the plan shrinks substantially.** What remains of the contracts phase
is: base-class contiguity check, `Snippet` read-only flag, delete `is_pure()` (base + catalog
sweep), delete `_validate_graph` and the channel-requirement machinery, delete `ErrorMode`,
simplify `CachePE`. Every item is a deletion or a one-liner.

### 5.3 Challenged — act on these

**The plan under-weights R5.** Its sequencing places the examples smoke test (E5) and CI (G4) in
Phase 2, behind the contract work. That is backwards. R5's gate is the countermeasure to the
root cause, it is cheap, and it has no dependency on the contract redesign. **Move E5 and G4 to
Phase 0.** Everything in Phase 1 is safer once the consumers are under test — and Phase 1's
blast radius is precisely what a consumer gate is for.

**The plan treats `CONTRIBUTING.md` as documentation.** Under R6 the new-PE checklist is a
*budget*, not prose — currently 7 steps, and the plan's F2 rewrite would add more. Bring it to 5
or push the difference into the base class.

---

## 6. Testing methodology

Four tiers, distinguished by *what kind of claim* each verifies. Under PD-2 the stakes rise:
with runtime validation deleted, **the test suite is the only systematic correctness check the
system has.** Tier 1 is no longer just the centerpiece of testing — it is the load-bearing wall
of the whole design.

**Current state, for calibration:** `tests/` is ~19,714 lines against ~19,660 in `src/` — a 1:1
ratio by volume — yet **28 modules have no test file at all.** Coverage is not thin; it is
wildly uneven. `conftest.py` is nine lines with a single autouse sample-rate fixture. There are
**zero golden or reference fixtures** of any kind. Assertions are overwhelmingly structural
(`shape`, `channels`, `duration`, coarse `np.all`/`np.allclose`). Nothing in the suite detects
*"it still runs, but it sounds wrong."*

### Tier 1 — Universal contract suite *(new; the centerpiece)*

One parametrised suite that **auto-discovers every PE from `__all__`** and asserts the base-class
contract against each. It must be auto-discovering, not a hand-maintained list — that property is
what makes a new PE inherit coverage for free, and is the direct expression of "easier to extend."

Claims it verifies, for every PE in the catalogue:

- `render(s, d)` returns exactly `d` samples starting at `s`
- returned `Snippet` is 2-D, `float32`, with the expected channel count, and its buffer is
  read-only
- samples outside `extent()` are zero-filled
- `extent()` is stable across the instance's lifetime
- `on_start()` / `on_stop()` are idempotent; `reset_state()` returns the PE to first-render state
- **randomised-order rendering**: pulled at shuffled `(start, duration)`, a PE either matches a
  contiguous reference or raises the contiguity error — never silently differs

The last line is what makes the contract falsifiable instead of prose, and is what would have
caught the `GainPE` case. It also covers all 28 currently-untested modules in one stroke.

*Registration is the gate:* a PE absent from Tier 1 discovery fails R7's export test.

### Tier 2 — Per-PE behavioural tests *(existing; keep, don't expand blindly)*

DSP correctness specific to one PE. This is what `tests/` mostly contains today. Tier 1 will
subsume some of it — delete what becomes redundant rather than carrying both.

**Rule for new tests:** contract-level tests go through a `Renderer`; only tests of pure
arithmetic may call `_render()` directly. Today ~830 direct `render()` calls versus 257
`NullRenderer` mentions means much of the suite bypasses the lifecycle it is meant to exercise.

### Tier 3 — Boundary tests *(new; R5's gate)*

Verifies that consumers of `src/` still work:

- every `examples/*.py` imports without error
- `benchmarks/benchmark_pes.py` runs
- README PE and example tables match `__all__` and the examples directory
- `import pygmu2` loads neither `scipy` nor `numba`, and stays within a stated cold-import budget

Cheap, headless, and collectively the countermeasure to §3 R5's evidence.

### Tier 4 — Numerical characterisation *(new; the "sounds wrong" detector)*

Scoped to the library layer, since the product is the catalogue, not the compositions. Assert
analytically checkable DSP properties rather than committing reference audio: THD for
band-limited oscillators, −3 dB points and rolloff slopes for filters, impulse-response length
for convolution, gain accuracy in dB for dynamics.

*Note:* `tests/test_analytical_pe.py` already exists as a seed for this tier — inspect before
building new infrastructure.

**Explicitly not adopted:** golden-audio regression over the etudes corpus. It is the right
centrepiece for a composition environment; the product is a library (§1), and golden WAVs are
brittle, large, and diagnose poorly.

### Migration note

DD-1 has a blast radius inside the test suite itself: **138 of ~830 `render()` calls use a
non-zero start.** Expect a meaningful fraction to need an explicit `reset_state()` or a renderer.
Triage them as the plan's A1 triage prescribes — each is either a legitimate seek or a latent
bug. Deleting `ErrorMode` similarly touches `tests/test_config.py` and every test that exercises
LENIENT behaviour.

---

## 7. Change protocol

The rule that prevents recurrence. This is the fourth deliverable, added to the original brief.

> **A change to `src/` is not complete until its consumers are proven.**

Concretely — a change may not land unless:

1. Tier 1 contract tests pass for every PE (not only ones you touched)
2. Tier 3 boundary tests pass — examples import, benchmarks run, docs tables match
3. Any renamed or removed public name has **zero remaining references** across `examples/`,
   `benchmarks/`, `scripts/`, `tests/`, and `*.md`
4. Any new invariant arrives with its gate in the same commit, or is not claimed at all (PD-1)
5. Any second-pass purchase (§4) arrives with the measured error or profile that motivates it

Point 3 is the direct countermeasure to §1's rename wave: pre-1.0 freedom (§1) means renaming is
permitted, but propagating it is not optional. Under §1's enforcement setting these are CI gates,
not review checklist items — a machine says no, so neither maintainer has to.

**Disabling a consumer is not a resolution.** Renaming a broken example to `.py-disabled` is how
eight examples, one benchmark suite, and three test files left the system unnoticed. Fix it or
delete it; there is no third state. *(Suggest CI fail on any new `*.py-disabled` file.)*

---

## 8. Enforcement matrix

Per §1, every rule is a gate. This is the Architect's build list.

| Rule | Gate | Home |
|---|---|---|
| PD-1 executable or deleted | Review question: "what test fails if this is violated?" | PR template line |
| PD-2 / §4 purchased complexity | Review question: "which measurement motivates this?" | PR template line |
| R1 facts not declarations | Tier 1 randomised-order test | `tests/test_contract.py` (new) |
| R2 loud late failure | Contiguity raise + read-only buffer cases | Tier 1 |
| R3 no silent decisions | Channel mismatch raises; grep: zero `ErrorMode`/`handle_error` refs | Tier 1 + CI grep |
| R4 one concept one home | Manifest check, naming-convention check, generated README tables | Tier 3 |
| R5 consumers verified | Example import test, benchmark smoke test | Tier 3 — **Phase 0** |
| R6 base class not template | 5-step budget check on the CONTRIBUTING checklist | CI lint on the doc |
| R7 public surface | Export-completeness walk over `src/pygmu2/*.py` | Tier 3 |
| Change protocol §7 | Zero-dangling-reference grep; no new `*.py-disabled` | CI |

Critical files: `src/pygmu2/processing_element.py` (R1, R2, R6 land here),
`src/pygmu2/renderer.py` (delete `_validate_graph`; keep lifecycle walkers),
`src/pygmu2/config.py` (delete `ErrorMode` machinery; keep `set_sample_rate`),
`src/pygmu2/audio_renderer.py` (lazy width in all playback paths),
`src/pygmu2/__init__.py` (R7), `tests/conftest.py` (Tier 1 discovery),
`.github/workflows/` (all gates).

---

## 9. What this permits you to stop doing

Explicitly licensed, so effort isn't spent defending debt:

- **Stop validating graphs.** `_validate_graph`, `required_input_channels()`,
  `resolve_channel_count()` — delete, don't refactor (PD-2, R2).
- **Stop preserving unreachable code.** `PortamentoPE`, `ExtentWindowPE`, and anything else not
  exported: export it or delete it (R7, §1 free rein).
- **Stop writing conventions you can't gate.** If it can't be checked, it isn't a rule (PD-1).
- **Stop maintaining two of anything** — profilers, manifests, naming schemes, playback paths,
  channel-width discovery (R4).
- **Stop hand-writing the README tables.** Generate them (R4).
- **Stop growing the new-PE checklist.** It has a budget of 5 (R6).
- **Stop building debugging aids and optimizations on speculation.** They are purchased with
  evidence (§4).
- **Stop treating `.py-disabled` as a state.** It isn't one (§7).

---

## 10. Verification

How the owner confirms this overhaul achieved its goal. Each is measurable against the audit
baseline, not a judgement call.

| Check | Command | Target |
|---|---|---|
| Contract coverage | `uv run pytest tests/test_contract.py -q` | Every PE in `__all__` parametrised; zero skips |
| Untested modules | module-vs-test-file walk | 28 → 0 for contract-level coverage |
| Consumers green | `uv run pytest tests/test_examples.py tests/test_boundaries.py` | All examples import; benchmarks run |
| Disabled files | `find . -name "*.py-disabled"` | 9 → 0 |
| Dangling old names | `grep -rE "RandomPE\|TriggerPE\|AdsrPE\|RampPE\|ResetPE" examples scripts benchmarks tests *.md` | Zero hits |
| Deleted machinery | grep `is_pure\|_validate_graph\|required_input_channels\|ErrorMode\|handle_error` in `src/` | Zero hits |
| Prose-only invariants | grep "framework enforces" and kin | Zero, or each has a named gate |
| Import cost | `uv run python -X importtime -c "import pygmu2"` | 740 ms → <100 ms; no scipy/numba |
| Extension cost | count steps in the CONTRIBUTING new-PE checklist | ≤5 |
| Core simplicity | `wc -l` on `processing_element.py` + `renderer.py` | **Lower than baseline** (343 + 583). The overhaul must shrink the core, not grow it |
| CI | push a branch renaming a public symbol without propagating | **Must fail.** This is the acceptance test for the whole overhaul |

The last two rows are the real ones. If the core got bigger, PD-2 was violated in spirit; if an
un-propagated rename can still land green, nothing structural has changed.
