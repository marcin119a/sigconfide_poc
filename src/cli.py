"""Command line entry point for a batch run."""

from __future__ import annotations

import argparse

import pandas as pd
from sigconfide.estimates.selection import hybrid_stepwise_selection
from sigconfide.io.spa_compat import fit_spa_style

_DEFAULTS = {
    k: v.default
    for k, v in __import__("inspect")
    .signature(hybrid_stepwise_selection)
    .parameters.items()
}


def _overdispersion(text):
    """None, "auto" (estimated per sample on the selected set) or a CV."""
    if text.lower() == "none":
        return None
    return "auto" if text.lower() == "auto" else float(text)


def _optional_float(text):
    return None if text.lower() == "none" else float(text)


def _margin(text):
    """A deviance margin, or 'bonferroni' for the panel-size rule."""
    return "bonferroni" if text.lower() == "bonferroni" else float(text)


def main(argv=None):
    p = argparse.ArgumentParser(prog="sigconfide")
    p.add_argument("--samples", required=True, help="contexts x samples TSV")
    p.add_argument("--signatures", required=True, help="contexts x signatures TSV")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument(
        "--anchors",
        nargs="*",
        default=[],
        metavar="SIG",
        help="anchor signatures by name (columns of --signatures), e.g. SBS1 SBS5: "
        "the search starts from them and never removes them",
    )
    p.add_argument(
        "--min-deviance-gain",
        type=_margin,
        default=_DEFAULTS["min_deviance_gain"],
        help="deviance margin a signature must clear (default %(default)s; "
        "'bonferroni' derives it from the number of signatures in the panel)",
    )
    p.add_argument(
        "--dispersion-scaled",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS["dispersion_scaled"],
        help=(
            "scale the margin by the dispersion of the support model "
            "(default %(default)s)"
        ),
    )
    p.add_argument(
        "--anchor-floor",
        type=_optional_float,
        default=_DEFAULTS["anchor_floor"],
        help="drop an anchor whose final exposure is below this fraction of the "
        "profile (default %(default)s; 'none' keeps every anchor)",
    )
    p.add_argument(
        "--anchor-min-cost",
        type=_optional_float,
        default=_DEFAULTS["anchor_min_cost"],
        help="also drop an anchor whose removal raises the deviance by less than "
        "this margin, scaled like --min-deviance-gain (default %(default)s = off; "
        "0 drops the anchors that only absorbed noise)",
    )
    p.add_argument(
        "--intervals",
        action="store_true",
        help="also write Assignment_Solution_Intervals.txt: bootstrap interval of "
        "every assigned exposure",
    )
    p.add_argument(
        "--R", type=int, default=100, help="bootstrap replicates for --intervals"
    )
    p.add_argument(
        "--level",
        type=float,
        default=0.90,
        help="interval level for --intervals (default 0.90)",
    )
    p.add_argument(
        "--overdispersion",
        type=_overdispersion,
        default="auto",
        help="over-dispersion of the bootstrap behind --intervals: 'auto' (default) "
        "estimates sigma_b per sample on the selected set, a number fixes it, "
        "'none' is the plain multinomial bootstrap",
    )
    p.add_argument("--n-jobs", type=int, default=None)
    a = p.parse_args(argv)

    anchor_indices = None
    if a.anchors:
        names = list(pd.read_csv(a.signatures, sep="\t", index_col=0, nrows=0).columns)
        unknown = [s for s in a.anchors if s not in names]
        if unknown:
            p.error(f"anchor signatures not in --signatures: {unknown}")
        anchor_indices = [names.index(s) for s in a.anchors]

    fit_spa_style(
        a.samples,
        a.signatures,
        a.output,
        R=a.R,
        n_jobs=a.n_jobs,
        intervals=a.intervals,
        overdispersion=a.overdispersion,
        level=a.level,
        anchor_indices=anchor_indices,
        min_deviance_gain=a.min_deviance_gain,
        dispersion_scaled=a.dispersion_scaled,
        anchor_floor=a.anchor_floor,
        anchor_min_cost=a.anchor_min_cost,
    )


if __name__ == "__main__":
    main()
