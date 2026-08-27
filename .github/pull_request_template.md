<!-- pygmu2 change protocol (DESIGN_PHILOSOPHY.md §7-§8). Answer both; delete whichever does not apply. -->

## What & why



## The two questions

**If this change claims an invariant — what test fails when it is violated?**
<!-- PD-1: executable or deleted. A convention without a gate is a suggestion. -->



**If this change adds a debugging aid or an optimization — which measured error or profile motivates it?**
<!-- PD-2: complexity is purchased with evidence, not presumed. Paste the
     hard-to-debug error, or the diagnostics.py profile naming the hot spot. -->



## Checklist

- [ ] `uv run pytest -q` green (contract suite covers every exported PE)
- [ ] Renamed/removed public names have zero remaining references in
      `examples/`, `benchmarks/`, `scripts/`, `tests/`, `*.md`
- [ ] New PE? Added to `__all__` and `tests/pe_factories.py` (contract
      coverage is inherited automatically)
- [ ] README tables regenerated if the public surface changed
      (`uv run python scripts/gen_readme_tables.py`)
