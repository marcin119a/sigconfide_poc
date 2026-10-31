# sigconfide

[![CI](https://github.com/wojtowicz-lab/sigconfide/actions/workflows/ci.yml/badge.svg)](https://github.com/wojtowicz-lab/sigconfide/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/wojtowicz-lab/sigconfide/branch/main/graph/badge.svg)](https://codecov.io/gh/wojtowicz-lab/sigconfide)

**sigconfide** is a lightweight Python library for estimating mutational
signature exposures and **selecting the active signatures** in a tumor sample.
Fitting a mutational profile to a signature panel (e.g. COSMIC) is solved as a
quadratic-programming (QP) problem, and the set of relevant signatures is chosen
with a stepwise (forward/backward) procedure whose significance is assessed via
bootstrap.

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
from sigconfide.estimates.selection import hybrid_stepwise_selection

# P: signature panel, shape (n_contexts, n_signatures)
# sig_names: signature names aligned with P's columns, e.g. ['SBS1', 'SBS2', ...]
# m: mutational profile for one sample, shape (n_contexts,) — raw counts are fine,
#    sigconfide normalizes internally to a distribution that sums to 1
sel_idx, exposures, errors = hybrid_stepwise_selection(
    m, P,
    R=100,                     # number of bootstrap replicates
    pre_filter_threshold=0.001 # drop trace signatures up front (speeds things up)
)

for idx, exp in zip(sel_idx, exposures):
    print(f"{sig_names[idx]}\t{exp:.3f}")
```

`hybrid_stepwise_selection` returns:
- `sel_idx` — indices of the selected signatures in the original panel `P`,
- `exposures` — relative exposures of the selected signatures (sum to 1),
- `errors` — reconstruction error (Frobenius norm).

Most important parameters:
- `R` — number of bootstrap replicates (more = more stable, slower),
- `mutation_count` — total number of mutations in the sample; needed only when `m` holds
  probabilities rather than integer counts, so the bootstrap knows how many mutations to
  resample,
- `min_support` (default `0.95`) — minimum bootstrap support (fraction of replicates in which
  a signature's exposure is positive) for a signature to be kept/added; this is a
  stability criterion with no exposure scale, not a significance test.
- `pre_filter_threshold` (default `0.001`) — a single fast QP solve prunes trace
  signatures before the bootstrap loop (~4× fewer signatures, no loss of sensitivity);
  pass `None` to disable it and search from the full panel,
- `anchor_indices` — indices of anchor signatures (e.g. the ubiquitous SBS1/SBS5): they
  survive the pre-filter, are present from the start and are never removed by the search;
  they face one check at the end, the `anchor_floor`,
- `anchor_floor` (default `0.002`) — relative exposure below which an anchor signature
  is dropped from the final answer after the search and the deviance steps. The anchor
  is still carried through the search, so a trace of SBS1 gets fitted at all and
  a look-alike cannot stand in for SBS5; the floor only asks whether the final fit used it.
  Columns below it go one at a time, smallest first, with a refit after each; at least two
  signatures survive. `None` keeps every anchor signature whatever its exposure,
- `min_deviance_gain` (default `10`) — after the bootstrap search, a swap step and a
  backward elimination scored by the Poisson deviance of the fit: an active signature is
  replaced by an inactive one when that lowers the deviance by more than this margin, and
  dropped when its removal raises it by less. The bootstrap asks whether an exposure is
  stable; this asks whether the signature is needed, and it keeps a peaked signature at a
  few percent of the profile while dropping a flat one parked on the residual.
  Anchor signatures are exempt; `None` disables the step,
- `dispersion_scaled` (default `True`) — puts `min_deviance_gain` on the quasi-likelihood
  scale: the margin is multiplied by the over-dispersion of the fit against the whole
  pre-filtered panel (`max(1, deviance/df)`), which leaves Poisson-or-quieter profiles
  unchanged and keeps the step meaningful on real catalogues, where the deviance per
  degree of freedom is typically 2--11; `False` applies the margin as given,
- `overdispersion` (default `None`) — over-dispersed bootstrap: each replicate first scales
  every channel by a gamma factor with this coefficient of variation, so a channel with `c`
  counts varies with SD `sqrt(c + (overdispersion * c)^2)` instead of `sqrt(c)`; on deep
  profiles this stops flat signatures from surviving on a residual the multinomial
  replicates cannot move. `"auto"` estimates it per sample from the residual of the fit
  against the pre-filtered panel (method of moments on `Var = c + sigma^2 c^2`, see
  `estimate_overdispersion`): it is 0, i.e. the plain multinomial bootstrap, when the
  residual is within counting noise, and the excess dispersion of the catalogue otherwise,
- `decomposition_method` (default `decomposeQP`) — solver `f(m, P) -> exposures` used for
  every fit, in the search and in the deviance steps alike,
- `max_iterations` (default `1000`) — hard cap on add/remove moves of the search, a
  last-resort guard against oscillation on degenerate low-count profiles,
- `rng` — source of the bootstrap draws: `None` (default) uses the global `np.random`, so
  `np.random.seed` controls the result; an int, a `SeedSequence` or a `Generator` gives an
  independent stream that leaves the global state alone. `bootstrapSigExposures` takes it too.

### Other functions in the package

```python
from sigconfide.estimates.standard    import findSigExposures          # fit exposures for ALL samples at once (matrix M 96×G)
from sigconfide.estimates.bootstrap   import bootstrapSigExposures     # bootstrap distribution of exposures for one sample
```

### Batch run (CLI)

```bash
python -m sigconfide.cli --samples samples.tsv --signatures signatures.tsv \
    --output out_dir [--R 100] [--n-jobs 4]
```

`samples.tsv` is contexts × samples, `signatures.tsv` is contexts × signatures
(first column = context names). The result is `out_dir/Assignment_Solution_Activities.txt`:
a TSV with samples as rows, signatures as columns and absolute mutation counts
as values (0 = not assigned). Python equivalent:
`sigconfide.io.spa_compat.fit_spa_style(...)`.

---

## 3. Repository layout

```
src/
  decompose/qp.py           # decomposeQP – QP solver (quadprog)
  estimates/
    standard.py             # findSigExposures
    bootstrap.py            # bootstrapSigExposures
    selection.py            # hybrid_stepwise_selection  ← main selection function
  io/spa_compat.py          # fit_spa_style – TSV in / Activities TSV out
  cli.py                    # python -m sigconfide.cli
  utils/utils.py            # FrobeniusNorm, is_wholenumber, resolve_rng
tests/
docs/                       # MkDocs sources; API reference is generated from docstrings
```
