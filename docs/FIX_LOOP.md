# Fix-loop declaration (one page)

> Fill in after the first full benchmark run (`python -m benchmark.evaluate`).
> Tag the commit the benchmark ran on as `fix-before` **before** writing any fix:
> `git tag fix-before`. Ship the fix as its own commits, then `git tag fix-after`.
> Everything regenerates with `python scripts/fix_loop.py`.

## 1. Worst-performing gate

- Gate: _e.g. opening widths (LiDAR tier)_
- Failing number: _e.g. 61% of openings within 2 cm (gate ≥ 85%)_
- Source: `benchmark/out/REPORT.md` at tag `fix-before`

## 2. Root-cause hypothesis and evidence

- Hypothesis: _one sentence_
- Evidence: _the specific rows/plots that point at it, and the check that rules out the obvious alternatives_

## 3. Fix and predicted number

- Fix: _what changes, in which file_
- Predicted gate value after the fix: _number, with the reasoning behind it_

## 4. Result (filled after `fix-after`)

- Measured gate value: _from `fixloop/GATES_DELTA.md`_
- Prediction vs result: _hit / short and why / overshoot and why_
- Side effects on other gates: _from the same table_

Artifacts: `fixloop/BEFORE_REPORT.md`, `fixloop/AFTER_REPORT.md`, `fixloop/GATES_DELTA.md`, `fixloop/code.diff`.
