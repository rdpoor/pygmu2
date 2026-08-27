# PE Catalog Consolidation Study

**Status:** Proposal for review — no code has changed. Implementation is planned separately,
after the owner reacts to the decision points in §F.
**Method:** Three parallel code explorations over the suspected overlap clusters
(extent/time, control-signal, oscillator/audio), reading every implicated PE's
implementation and taking a usage census across `src/`, `tests/`, `examples/`, `scripts/`,
`benchmarks/`. 2026-08-26.
**Ground rules** (per `DESIGN_PHILOSOPHY.md`): when a merge lands, old names are deleted
outright — no aliases, no dual vocabulary. Every claim below cites the exact delta a reader
can verify by diffing the named files.

## Summary

The catalog has 71 concrete PEs. This study proposes a target of **64**, via six
merges/deletions plus one demotion — every one a case where two PEs differ by a handful of
lines or one is dead. Along the way the exploration surfaced **9 bugs/dead-code findings**
(§D) and **5 contract ambiguities** (§E) worth fixing regardless of consolidation.

The catalog's real problem is narrower than "too many PEs": it is **five pairs of
near-twins** (one-line or one-method deltas), a **gate/trigger family whose duality is
nominal rather than mechanical**, and a handful of contracts that say one thing while the
code does another.

---

## A. Strong merges (near-duplicates)

### A1. `CropPE` + `SetExtentPE` → `CropPE`

Both are thin constructors over the shared `_ExtentWindowPE` base — they contribute **zero**
rendering code of their own, and `CropPE(src, 10, 5)` and `SetExtentPE(src, 10, 5)` render
byte-identical output for every request. The entire delta:

| | CropPE | SetExtentPE |
|---|---|---|
| `extent()` | window ∩ source — can only shrink | window verbatim — can pad past the source |
| `start=None` | rejected (via `int(None)` TypeError) | allowed — but then `duration` silently means *absolute end*, the murkiest contract in the family |

Usage: CropPE has 300+ call sites (the universal "bound this to N seconds" idiom).
SetExtentPE has exactly **2 real ones** — `tralfam_pe.py:60` (zero-padding) and
`examples_helper.pad_clip` (trailing silence) — both using the one capability CropPE lacks.

**Merged contract:**

```python
CropPE(source, start: int, duration: int | None, *,
       extend_mode=ExtendMode.ZERO, clip: bool = True)
```

`clip=False` gives today's SetExtentPE extent (pad past the source). `start=None` support is
dropped — nothing real uses it, and its `duration`-means-`end` overload should not survive.
SetExtentPE is deleted; its 2 call sites become `CropPE(..., clip=False)`.

### A2. `AnalogOscPE` + `FunctionGenPE` → `AnalogOscPE`

These are the same PE twice: ~110 of ~200 substantive lines duplicated, byte-identical
validation strings, identical class constants, lifecycle, phase accumulation, and channel
expansion. The differences: FunctionGen has a `phase` parameter and naive (aliased) waveform
math with default `frequency=1.0` (LFO-oriented); AnalogOsc adds polyBLEP antialiasing and
clamps `duty` away from 0/1 — which makes its documented "duty=0 → saw" claim **false**;
only FunctionGen delivers an exact saw.

**Merged contract:**

```python
AnalogOscPE(frequency=440.0, duty_cycle=0.5, phase=0.0,
            waveform="rectangle", antialias=True, channels=1)
```

`antialias=False` is today's FunctionGenPE. Document the cliff: `antialias=True` clamps duty
to `[2·dt, 1−2·dt]`, so exact saw-up/down requires `antialias=False`. FunctionGenPE is
deleted; `periodic_gate.py` (its one internal consumer — a gate *needs* naive hard edges)
passes `antialias=False`; `function_gen_aliasing_eg.py` becomes a better demo — one class,
one flag A/B. Implementation care point: the phase offset must be applied *before* the BLEP
residual computation, not after.

### A3. `RandomValuePE` + `RandomStepPE` → `RandomValuePE`

The render loops differ in **one line** — smooth one-pole toward a random target vs. jump to
it instantly. Everything else (rate handling, seeding, lifecycle, statefulness, extent) is
byte-identical.

**Merged contract:**

```python
RandomValuePE(rate=10.0, seed=None, smoothing: float | None = None)
```

`smoothing=1.0` is today's RandomStepPE; `None` (auto: `rate/sr`) is today's RandomValuePE.
Decoupling `smoothing` from `rate` is **strictly more expressive** than either PE today —
slow wander with fast jumps (or the reverse) becomes possible, which the hard-wired coupling
currently forbids. RandomStepPE is deleted. RNG draw order is preserved, so seeded streams
stay bit-exact. (Do not name the merged PE `RandomPE` — that name is on the
dangling-old-names grep gate.)

Side observation: `RandomStepPE ≡ SampleHoldPE(noise, RandomTriggerPE(rate))` — three
spellings of one idea today, none documented as equivalent.

### A4. `SampleHoldPE` + `TrackHoldPE` → `HoldPE`

91 lines each; the entire behavioral difference is **one comparison inside the loop**:
`trig[i] > 0` (latch on event) vs `gate[i] > 0.5` (latch while high).

**Merged contract:**

```python
HoldPE(source, control, mode=HoldMode.SAMPLE | HoldMode.TRACK, initial_value=0.0)
```

Both old names are deleted. The merge fixes two shared warts in the same stroke: normalize
both thresholds to `> 0` (GateSignal guarantees exactly 0/1, so `> 0.5` is superstition) and
propagate the source extent instead of hardcoding infinite.

---

## B. The gate/trigger family: make gate the primitive

The `GateSignal`/`TriggerSignal` duality is nominal, not mechanical:

- **`PeriodicGate` and `PeriodicTrigger` run on different timebases** — continuous
  fractional phase (via an internal FunctionGenPE) vs. integer-quantized period
  (`int(round(sr/hz))`) — and therefore **drift apart** at non-integer `sr/hz`. Meanwhile
  `examples/adsr_eg.py` explicitly relies on them staying in sync, and carries a stale TODO
  wishing for a gate→trigger converter — **`GateToTriggerPE` already exists**, is exported,
  and has a 262-line test file. The redundancy is already confusing the project's own
  authors.
- **`RandomGatePE` and `RandomTriggerPE` are the same file minus 3 lines** — with a hidden
  2× rate trap: the gate *toggles* per event, so a trigger derived from it fires at half the
  rate.
- The trigger contract's falling-edge and multiplicity clauses are **dead**: no subclass
  emits negatives, every consumer tests `> 0`, and `PeriodicTrigger(amplitude=3)` means
  "three simultaneous events" per the ABC but "amplitude 3" per its own docstring.
- The generator matrix has a hole: `ScheduledGatePE` **silently merges overlapping
  intervals**, so two legato notes become one gate → one trigger instead of two — a real
  correctness gap for `AdsrTriggeredPE`.

**Proposal — gates are the primitives; triggers are derived:**

1. Delete **`PeriodicTrigger`** → `GateToTriggerPE(PeriodicGate(frequency=f))`. One
   timebase; the drift bug disappears structurally. Costs: the stateless index-based fast
   path (acceptable per PD-2 — no pre-optimization) and the contract-ambiguous `amplitude`.
2. Delete **`RandomTriggerPE`** → `GateToTriggerPE(RandomGatePE(rate=2*r))`. **The 2× rate
   convention is the most dangerous migration item in this study** and must be documented
   loudly at the call sites.
3. **`ScheduledGatePE`** gains `merge_overlaps: bool = True` so onsets can survive
   conversion. With that, no `ScheduledTriggerPE` is needed — the composition is correct.
4. **Tighten the ABCs** so the distinction is stated, not folklore: GateSignal = sustained
   level, exactly {0, 1}, duration meaningful; TriggerSignal = isolated one-sample events
   (a run of length > 1 is a contract violation). Drop the dead sign/multiplicity clauses.
   Factor the duplicated `_env_flag`/validation scaffolding into one shared base.
5. **Do not add `TriggerToGatePE`.** Nothing needs it; the asymmetry is accepted (gate is
   the primitive). Second-pass discipline: build it when something asks for it.

---

## C. Keep separate — lookalikes that aren't duplicates

| Pair | Why they stay separate |
|---|---|
| `TimeWarpPE` vs `ResamplePE` | Genuinely different contracts despite identical surface: TimeWarp **integrates** a persisted head position (stateful, supports negative/zero/PE rates, ignores the request's absolute position); Resample maps **absolute position** (stateless/seekable — which is exactly why `NotesPE` uses it for N parallel notes). A merge would flip statefulness, seekability, extent formula, and legal rate domain on one boolean. Fix: document the positional-vs-integrated distinction in both docstrings. |
| `BlitSawPE` vs `AnalogOscPE` | Different algorithms, disjoint controls (`m` harmonic count, `leak`, `amplitude`, `initial_phase` vs duty/waveform morph). `SuperSawPE` builds on BlitSaw. |
| `RingModulatorPE` vs `GainPE` | The composition equivalent is a 5-node graph plus a `CachePE` the user would forget (RingMod renders its carrier once). |
| `ControlPE` vs `ConstantPE` | Purity is the load-bearing difference: settable-from-any-thread impure source vs immutable constant. Keep both — but **rename `ControlPE`**, which collides with the "control signal" category (candidate: `SettableValuePE`; §F4). |
| `WavReaderPE` vs `AudioReaderPE` | Core-dep per-block streaming vs extra-dep whole-file decode — a real axis. But **their sample-rate contracts contradict**: WavReader *warns and plays at the wrong pitch* on a rate mismatch (a silent decision, R3) while AudioReader silently resamples. Proposal: WavReaderPE raises on mismatch (the SpatialHRTF precedent from the overhaul); AudioReaderPE documents that it resamples. |
| `Compressor/Limiter/Expander` vs `DynamicsPE` | Thin presets (~12–28 lines) but valuable ones — they encode the `CachePE` fan-out footgun a hand-rolled graph would forget. Keep, minus the dead code in §D3. |
| `DecayingSinePE` vs `GainPE(SinePE, env)` | Keep — its real value is the **finite extent known at construction** (the composition's extent is infinite without an extra Crop). But see §D7. |

---

## D. Bugs and dead code found along the way (worth fixing regardless)

1. **`DelayPE` out-of-bounds bug:** half-bounded sources (e.g. `Extent(0, None)`) skip the
   OOB mask entirely (`and` where its siblings use `or`), leaking edge-clamped samples.
   Fix by unifying the three divergent OOB-mask implementations (delay/timewarp/resample)
   into one helper in `interpolated_lookup.py`.
2. **`SuperSawPE` dead `CachePE`** (`super_saw_pe.py:228-232`): built and never used, so a
   PE-valued frequency renders once *per voice* per block — a real perf bug. Also line 249
   passes a length-1 ndarray (not a float) as `initial_phase`.
3. **Dead dynamics code:** `DynamicsMode.LIMIT` is unreachable (LimiterPE uses `ratio=100`);
   `DynamicsMode.EXPAND` + `_compute_expansion_gain` have no caller (ExpanderPE uses GATE);
   `in_knee` locals are computed and never read.
4. **`WavReaderPE.native_rate`:** dead, zero references anywhere.
5. **Dead non-contiguous guards** in AnalogOscPE/FunctionGenPE — the base class raises
   before they can trigger.
6. **`PeriodicGate` hides its internal FunctionGenPE** from `inputs()` (returns the child's
   inputs and hand-forwards lifecycle) — the same anti-pattern IdiophonePE had. Expose it.
7. **`DecayingSinePE`'s "optimization" is a Python for-loop** — slower than the vectorized
   closed form (`amp · ρⁿ · sin(nω)`) it was meant to beat. Vectorize; keep the PE.
8. **`_ExtentWindowPE` channel inconsistency:** can emit 1-channel silence outside the
   window and N-channel audio inside it when `channel_count()` is None; its hold-value
   cache swallows all exceptions and caches forever.
9. **`test_set_extent_pe.py` HOLD_FIRST test cannot fail** — it asserts zeros against a
   source whose first value is 0.0.

## E. Contract ambiguities to resolve (documentation-level)

1. **`extend_mode` hold regions live *outside* the advertised `extent()`** family-wide —
   the feature is only observable when a wider consumer pulls past the extent. Say so
   plainly in the docstrings (folding holds into the extent would make HOLD_LAST infinite —
   probably undesirable).
2. **The PE-valued-modulator extent rule differs per PE:** DelayPE intersects
   source ∩ delay; TimeWarpPE uses the rate's extent alone. Pick one convention.
3. **Channel broadcast is solved twice, differently:** GainPE `np.tile`s (copies),
   RingModulator `np.broadcast_to`s (views); neither validates the stereo-modulator/
   mono-carrier direction. One shared helper.
4. **Naming:** `PeriodicGate` (and formerly `PeriodicTrigger`) lack the `PE` suffix every
   sibling carries.
5. **`IdentityPE` is a test fixture promoted to the public API:** 0 examples, 0 src usage;
   11 test files use it as a timestamp probe. Either demote it to `tests/`, or keep it
   public and land the missing wavetable example that would justify the
   ArrayPE/IdentityPE/WavetablePE trio — whose docstring currently advertises two PEs
   (`PhaseAccumulatorPE`, `RandomPE`) that don't exist.

## F. Decision points for the owner

1. **RandomTriggerPE deletion:** accept the 2× rate-convention migration, or keep it?
2. **PeriodicTrigger deletion:** accept losing the stateless fast path and `amplitude`?
3. **IdentityPE:** demote to tests, or keep public + write the wavetable example?
4. **ControlPE rename:** `SettableValuePE`? another name?
5. **WavReaderPE rate mismatch:** raise (recommended, R3) or resample like AudioReaderPE?

## G. Catalog arithmetic

Deletions: `FunctionGenPE`, `SetExtentPE`, `RandomStepPE`, `PeriodicTrigger`,
`RandomTriggerPE`, `SampleHoldPE`+`TrackHoldPE`→`HoldPE` (net −1), `IdentityPE` (if
demoted) → **71 → 64 concrete PEs.**

Safety net: every name above is registered in `tests/pe_factories.py` (the single biggest
blast-radius file), so every merge is exercised by the parametrized contract suite —
statefulness, reset, extent, and ordering regressions are caught automatically. Seeded-
stream tests need regeneration only where RNG draw order changes (A3 preserves it
deliberately).

## H. Implementation sketch (for the later pass)

Cheapest and safest first:

1. Fix `adsr_eg.py`'s stale TODO with the existing `GateToTriggerPE` — zero risk, and it
   demonstrates the derivation path the B-proposal builds on.
2. §D bug fixes and §E docstring tightenings.
3. A4 `HoldPE` (one comparison).
4. A3 `RandomValuePE` (one line).
5. A1 `CropPE` absorb (extent metadata only).
6. A2 oscillator merge (care point: phase-before-BLEP).
7. §B trigger deletions — real behavioral change, last.

Each step lands under the change protocol: contract suite green, zero dangling references,
README tables regenerated.
