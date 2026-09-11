#!/bin/zsh
# Rebuild the COSMIC v3.6 benchmark set end to end and score it.
#
#   1. make_diazgay_v36.py        3600 synthetic tumours (12 PCAWG types x 300),
#                                 SynSigGen recipe on the PCAWG attribution,
#                                 catalogues from the COSMIC v3.6 panel
#   2. benchmark_isbs.py x 3      SPA (needs .venv-spa), sigconfide at defaults,
#                                 sigconfide with SBS1/SBS5 forced; restricted
#                                 and full ground truth, clean/noise5/noise10
#   3. make_diazgay_v36_tables.py comparison.csv / comparison.md in the set,
#                                 diazgay_v36_micro_f1.csv next to the README
#
# Inputs: PCAWG_Benchmark/published/ (fetch_pcawg_benchmark.py) and
# tests/data/COSMIC_v3.6_SBS_GRCh37.txt.  Runs the arms one after another;
# about 25 minutes on 14 cores.  Pass --seed N and/or --out-dir DIR to build
# a different draw (defaults: seed 0, PCAWG_Benchmark/diazgay-v36).
set -euo pipefail
cd "$(dirname "$0")"

SEED=0
DIR=PCAWG_Benchmark/diazgay-v36
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --out-dir) DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PY=.venv/bin/python
PY_SPA=.venv-spa/bin/python
[[ -x $PY ]] || { echo "missing $PY" >&2; exit 1; }

echo "== 1/3 generate ($DIR, seed $SEED) =="
$PY make_diazgay_v36.py --seed "$SEED" --out-dir "$DIR"

echo "== 2/3 score =="
rm -rf "$DIR/eval"
if [[ -x $PY_SPA ]]; then
  $PY_SPA benchmark_isbs.py --benchmark-dir "$DIR" --method spa --label spa \
      --ground-truth restricted full --append
else
  echo "no .venv-spa, skipping the SPA arm" >&2
fi
$PY benchmark_isbs.py --benchmark-dir "$DIR" --label sc \
    --ground-truth restricted full --append
$PY benchmark_isbs.py --benchmark-dir "$DIR" --label sc_mandatory --mandatory SBS1 SBS5 \
    --ground-truth restricted full --append

echo "== 3/3 tables =="
$PY make_diazgay_v36_tables.py --set-dir "$DIR"
