"""Command line entry point for a batch run."""

from __future__ import annotations

import argparse

from sigconfide.io.spa_compat import fit_spa_style


def main(argv=None):
    p = argparse.ArgumentParser(prog="sigconfide")
    p.add_argument("--samples", required=True, help="contexts x samples TSV")
    p.add_argument("--signatures", required=True, help="contexts x signatures TSV")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--R", type=int, default=100, help="bootstrap replicates")
    p.add_argument("--n-jobs", type=int, default=None)
    a = p.parse_args(argv)
    fit_spa_style(a.samples, a.signatures, a.output, R=a.R, n_jobs=a.n_jobs)


if __name__ == "__main__":
    main()
