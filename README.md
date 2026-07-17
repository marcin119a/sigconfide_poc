# sigconfide

[![CI](https://github.com/marcin119a/sigconfide_poc/actions/workflows/ci.yml/badge.svg)](https://github.com/marcin119a/sigconfide_poc/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/marcin119a/sigconfide_poc/branch/main/graph/badge.svg)](https://codecov.io/gh/marcin119a/sigconfide_poc)

**sigconfide** is a lightweight Python library for estimating mutational
signature exposures and **selecting the active signatures** in a tumor sample.
Fitting a mutational profile to a signature panel (e.g. COSMIC) is solved as a
quadratic-programming (QP) problem, and the set of relevant signatures is chosen
with a stepwise (forward/backward) procedure whose significance is assessed via
bootstrap.

This README shows how to run sigconfide on the bundled **example SBS data**
(single base substitutions) from the Diaz-Gay et al. 2023 benchmark.

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

## 2. Example SBS data

The data lives in
`Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS/`:

| File | Format | Contents |
|------|--------|----------|
| `COSMIC_v3.3_SBS_GRCh37.txt` | TSV, `index_col=0` | Signature panel: **96 contexts × 78 signatures** (SBS1, SBS2, …) |
| `Samples.txt` | TSV, `index_col=0` | Mutational profiles (mutation counts): **96 contexts × 2700 samples** (clean data) |
| `Samples_noise5.txt` / `Samples_noise10.txt` | TSV | The same samples with 5% / 10% noise |
| `ground.truth.syn.exposures.csv` | CSV, `index_col=0` | True exposures: **signatures × samples** (for validation) |

Key conventions:
- Rows are the **96 SBS contexts** (e.g. `A[C>A]A`); before computing, sigconfide
  aligns the panel and sample rows by this index.
- Columns of `Samples.txt` are individual samples (`SP.Syn.Bladder-TCC::S.1`, …).
- The input vector `m` may contain **raw mutation counts** — sigconfide normalizes
  it internally to a distribution that sums to 1.

---

## 3. Quick start — ready-made script

The repository ships a ready example, [`examples/run_sbs_example.py`](examples/run_sbs_example.py),
which loads the SBS data, runs signature selection for a single sample, and
compares the result against the ground truth:

```bash
# first sample, clean data
python examples/run_sbs_example.py

# a specific sample by name
python examples/run_sbs_example.py "SP.Syn.Bladder-TCC::S.1"

# sample by index, data with 5% noise
python examples/run_sbs_example.py --sample-index 5 --noise noise5
```

Example output:

```
Sample:            SP.Syn.Bladder-TCC::S.1  (noise: clean)
Signature panel:   78 COSMIC SBS signatures
Mutation count:    ...

Detected signatures (relative exposure):
  SBS13     0.420
  SBS2      0.310
  SBS1      0.150
  ...

Comparison with ground truth:
  precision=... recall=...
```

---

## 4. Using the API

Minimal code that does the same thing as the script above:

```python
import numpy as np
import pandas as pd
from sigconfide.estimates.selection import hybrid_stepwise_selection

BASE = "Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/SBS"

# 1. Load the signature panel and the mutational profiles
cosmic  = pd.read_csv(f"{BASE}/COSMIC_v3.3_SBS_GRCh37.txt", sep="\t", index_col=0)
samples = pd.read_csv(f"{BASE}/Samples.txt",                sep="\t", index_col=0)

# 2. Align the 96 mutation contexts between the panel and the samples
idx     = cosmic.index.intersection(samples.index)
P       = cosmic.loc[idx].values.astype(float)     # signature matrix (96 x 78)
sig_names = np.array(cosmic.columns)               # ['SBS1', 'SBS2', ...]

# 3. Pick one sample (raw mutation counts – they will be normalized)
m = samples.loc[idx, "SP.Syn.Bladder-TCC::S.1"].values.astype(float)

# 4. Select the active signatures + estimate their exposures
sel_idx, exposures, errors = hybrid_stepwise_selection(
    m, P,
    R=100,                     # number of bootstrap replicates
    pre_filter_threshold=0.001 # drop trace signatures up front (speeds things up)
)

for name, exp in zip(sig_names[sel_idx], exposures):
    print(f"{name}\t{exp:.3f}")
```

`hybrid_stepwise_selection` returns:
- `sel_idx` — indices of the selected signatures in the original panel `P`,
- `exposures` — relative exposures of the selected signatures (sum to 1),
- `errors` — reconstruction error (Frobenius norm).

Most important parameters:
- `R` — number of bootstrap replicates (more = more stable, slower),
- `significance_level` (default `0.05`) — significance threshold for adding/removing signatures,
- `pre_filter_threshold` — if set (e.g. `0.001`), a single fast QP solve prunes trace
  signatures before the bootstrap loop (~4× fewer signatures, no loss of sensitivity),
- `mandatory_indices` — indices of signatures that are always present (e.g. the ubiquitous
  SBS1/SBS5) and are never removed.

### Other functions in the package

```python
from sigconfide.estimates.standard    import findSigExposures          # fit exposures for ALL samples at once (matrix M 96×G)
from sigconfide.estimates.bootstrap   import bootstrapSigExposures     # bootstrap distribution of exposures for one sample
from sigconfide.estimates.crossvalidation import crossValidationSigExposures  # cross-validation
```

---

## 5. Full benchmark

To reproduce the full precision/recall/F1 evaluation over the whole SBS set
(and optionally DBS/ID/CN):

```bash
# SBS only, all noise levels
python benchmark_diaz_gay.py --mut-types SBS

# quick test on 20 samples, clean data
python benchmark_diaz_gay.py --mut-types SBS --noise-levels clean --max-samples 20
```

Results are written to `benchmark_diaz_gay_results.csv` (per sample) and
`benchmark_diaz_gay_summary.csv` (aggregates). The benchmark uses
multiprocessing, so a full run (2700 samples) takes a while — for a quick
sanity check use `--max-samples`.

---

## 6. Repository layout

```
src/
  decompose/qp.py           # decomposeQP – QP solver (quadprog)
  estimates/
    standard.py             # findSigExposures
    bootstrap.py            # bootstrapSigExposures
    selection.py            # hybrid_stepwise_selection  ← main selection function
  utils/utils.py            # FrobeniusNorm, is_wholenumber
examples/
  run_sbs_example.py        # the example from this README
benchmark_diaz_gay.py       # full benchmark (SBS/DBS/ID/CN)
compare_spa.py              # comparison against SigProfilerAssignment
Supplementary_data_Diaz-Gay_et_al_2023_Benchmark/   # input data
```
