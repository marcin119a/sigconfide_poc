# sigconfide

[![CI](https://github.com/wojtowicz-lab/sigconfide/actions/workflows/ci.yml/badge.svg)](https://github.com/wojtowicz-lab/sigconfide/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/wojtowicz-lab/sigconfide/branch/main/graph/badge.svg)](https://codecov.io/gh/wojtowicz-lab/sigconfide)

**sigconfide** is a lightweight Python library for estimating mutational
signature exposures and **selecting the active signatures** in a tumor sample.
Fitting a mutational profile to a signature panel (e.g. COSMIC) is solved as a
quadratic-programming (QP) problem, and the set of relevant signatures is chosen
with a stepwise (forward/backward) procedure whose significance is assessed via
bootstrap.

> Benchmark scripts, example/benchmark data, and comparisons against other
> refitters (SigProfilerAssignment, MuSiCal) live in the sibling
> `results_sigconfide/` directory at the monorepo root — see the top-level
> README there for how to run them.

---

## 1. Installation

The project requires **Python ≥ 3.10**. A virtual environment is recommended:

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
- `min_support` (default `0.95`) — minimum bootstrap support (fraction of replicates in which
  a signature's exposure exceeds `threshold`) for a signature to be kept/added; this is a
  stability criterion, not a significance test,
- `pre_filter_threshold` (default `0.001`) — a single fast QP solve prunes trace
  signatures before the bootstrap loop (~4× fewer signatures, no loss of sensitivity);
  pass `None` to disable it and search from the full panel,
- `mandatory_indices` — indices of signatures that are always present (e.g. the ubiquitous
  SBS1/SBS5) and are never removed,
- `rng` — source of the bootstrap draws: `None` (default) uses the global `np.random`, so
  `np.random.seed` controls the result; an int, a `SeedSequence` or a `Generator` gives an
  independent stream that leaves the global state alone. `bootstrapSigExposures` takes it too.

For a runnable end-to-end example loading real COSMIC panels and sample data,
see `results_sigconfide/examples/run_sbs_example.py` in the monorepo root.

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

Benchmarks, example/comparison scripts and their input data are not part of
this package — they live in `results_sigconfide/` at the monorepo root.
