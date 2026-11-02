# sigconfide

[![CI](https://github.com/wojtowicz-lab/sigconfide/actions/workflows/ci.yml/badge.svg)](https://github.com/wojtowicz-lab/sigconfide/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/wojtowicz-lab/sigconfide/branch/main/graph/badge.svg)](https://codecov.io/gh/wojtowicz-lab/sigconfide)

**sigconfide** is a lightweight Python library for estimating mutational
signature exposures and **selecting the active signatures** in a tumor sample.
Fitting a mutational profile to a signature panel (e.g. COSMIC) is solved as a
quadratic-programming (QP) problem; the set of active signatures is chosen by a
deterministic forward-backward search on the Poisson deviance of the fit, and the
bootstrap reports the uncertainty of the selected exposures (intervals and
selection stability).

---

## 1. Installation

The project requires **Python ≥ 3.11**. A virtual environment is recommended:

```bash
# create and activate a venv
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# install the package in editable mode together with its dependencies
pip install -e .

# optional: plotting / notebook extras (matplotlib, ipykernel)
pip install -e ".[dev]"
```

Runtime dependencies: `numpy`, `quadprog`, `pandas`, `scipy`.

Check that the package imports:

```bash
python -c "from sigconfide.estimates.selection import hybrid_stepwise_selection; print('OK')"
```

---

## 2. Using the API

`hybrid_stepwise_selection` takes a mutational profile `m` (raw counts, one
sample) and a signature panel `P` (contexts × signatures), and returns the
selected signature subset with their exposures:

```python
import numpy as np
from sigconfide.estimates.selection import hybrid_stepwise_selection, exposure_intervals

# P: signature panel, shape (n_contexts, n_signatures)
# sig_names: signature names aligned with P's columns, e.g. ['SBS1', 'SBS2', ...]
# m: mutational profile for one sample, shape (n_contexts,) — raw counts
anchors = [sig_names.index("SBS1"), sig_names.index("SBS5")]
sel_idx, exposures, errors = hybrid_stepwise_selection(m, P, anchor_indices=anchors)

ci = exposure_intervals(m, P, sel_idx, B=100)          # bootstrap layer, 90 % intervals
for idx, exp, lo, hi in zip(sel_idx, exposures, ci["lower"], ci["upper"]):
    print(f"{sig_names[idx]}\t{exp:.3f}\t[{lo:.3f}, {hi:.3f}]")
```

`hybrid_stepwise_selection` returns:
- `sel_idx` — indices of the selected signatures in the panel `P`,
- `exposures` — relative exposures of the selected signatures (sum to 1),
- `errors` — reconstruction error (Frobenius norm).

How it selects. A first QP against the whole panel gives the saturated fit; the
columns it uses are the largest identifiable model and give the quasi-Poisson
dispersion `phi = deviance / (K - p)` of the catalogue. The search then starts from
the anchor signatures (or the two strongest columns of the saturated fit) and, at each
step, applies the best single move over the whole panel: a signature enters when it
lowers the Poisson deviance of the fit by at least the margin
`min_deviance_gain * max(1, phi)`, and leaves when it accounts for less than that
margin. It stops when no move qualifies. No resampling is involved, so the result is
deterministic and takes a few milliseconds per sample. The margin `10` is the gain an
absent signature exceeds with probability 0.0016 under Poisson noise (`chi2_1`), about
the 95th percentile of the largest gain among a hundred absent candidates and about
`log n` for a few thousand mutations; `phi` keeps that meaning on over-dispersed
catalogues.

Parameters:
- `mutation_count` — total number of mutations in the sample; needed only when `m`
  holds probabilities rather than integer counts,
- `anchor_indices` — indices of anchor signatures (e.g. the ubiquitous SBS1/SBS5): the
  search starts from them and never removes them; they face one check at the end,
- `anchor_floor` (default `0.002`) — relative exposure below which an anchor signature is
  dropped from the final answer: the anchor is carried through the search so that a trace
  of SBS1 is fitted at all and a look-alike cannot stand in for SBS5; this last step asks
  whether the final fit actually used it. Columns below the floor go one at a time,
  smallest first, with a refit after each; at least two signatures survive. `None` keeps
  every anchor,
- `anchor_min_cost` (default `None`, off) — second condition of that check: an anchor is
  also dropped when removing it from the final set raises the Poisson deviance by less
  than this margin (scaled by `max(1, phi)` like the search margin). `0` drops the
  anchors whose removal does not raise the deviance at all (they only absorbed noise)
  and keeps every anchor the fit relies on, however small,
- `min_deviance_gain` (default `10`) — the deviance margin `gamma_0` a signature must
  clear, before the dispersion scaling; `"bonferroni"` derives it from the size of the
  panel (`bonferroni_margin(P.shape[1])`: the 95th percentile of the largest gain among
  the panel's absent candidates under Poisson noise, 7.0 for 12 signatures, 8.0 for 21,
  10.8 for 101),
- `dispersion_scaled` (default `True`) — multiply the margin by `max(1, phi)`, the
  dispersion of the saturated fit; `False` applies `min_deviance_gain` as given,
- `decomposition_method` (default `decomposeQP`) — solver `f(m, P) -> exposures`,
- `max_iterations` (default `200`) — cap on the number of moves.

Uncertainty of the selected exposures (the bootstrap layer):
- `exposure_intervals(m, P, sel_idx, B=100, overdispersion="auto", level=0.90, rng=None)`
  refits the selected set on `B` replicates of the profile and returns the central
  `level` interval, the SD and the replicates. With `overdispersion="auto"` the
  replicates are over-dispersed by the `sigma_b` estimated on the selected set
  (`estimate_overdispersion(..., leverage=True)`: `Var = c + (sigma_b c)^2` per channel,
  0 when the residual is within counting noise); a number fixes `sigma_b`, `None` is the
  plain multinomial bootstrap,
- `selection_stability(m, P, B=100, ...)` runs the whole selection on each replicate and
  returns, per column of `P`, the fraction of replicates in which that signature is
  selected,
- `rng` — `None` uses the global `np.random`; an int, a `SeedSequence` or a `Generator`
  gives an independent stream.

### Other functions in the package

```python
from sigconfide.estimates.standard    import findSigExposures          # fit exposures for ALL samples at once (matrix M 96×G)
from sigconfide.estimates.bootstrap   import bootstrapSigExposures     # bootstrap distribution of exposures for one sample
```

### Batch run (CLI)

```bash
python -m sigconfide.cli --samples samples.tsv --signatures signatures.tsv \
    --output out_dir [--anchors SBS1 SBS5] [--intervals --R 100] [--n-jobs 4]
```

`samples.tsv` is contexts × samples, `signatures.tsv` is contexts × signatures
(first column = context names). The result is `out_dir/Assignment_Solution_Activities.txt`:
a TSV with samples as rows, signatures as columns and absolute mutation counts
as values (0 = not assigned). With `--intervals` a second file,
`Assignment_Solution_Intervals.txt`, holds one row per assigned signature with the
bootstrap interval of its exposure (`--R` replicates, `--level`, `--overdispersion`).
Python equivalent: `sigconfide.io.spa_compat.fit_spa_style(...)`.

---

## 3. Repository layout

```
src/
  decompose/qp.py           # decomposeQP – QP solver (quadprog)
  estimates/
    standard.py             # findSigExposures
    bootstrap.py            # bootstrapSigExposures
    selection.py            # hybrid_stepwise_selection, exposure_intervals, selection_stability
  io/spa_compat.py          # fit_spa_style – TSV in / Activities TSV out
  cli.py                    # python -m sigconfide.cli
  utils/utils.py            # FrobeniusNorm, is_wholenumber, resolve_rng
tests/
docs/                       # MkDocs sources; API reference is generated from docstrings
```
