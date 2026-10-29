import numpy as np
import pandas as pd
import pytest
from sigconfide.cli import _overdispersion, main
from sigconfide.io.spa_compat import OUTPUT_NAME, read_activities


def test_overdispersion_flag_parsing():
    assert _overdispersion("auto") == "auto"
    assert _overdispersion("none") is None
    assert _overdispersion("0.1") == pytest.approx(0.1)
    with pytest.raises(ValueError):
        _overdispersion("lots")


def test_cli_runs_with_auto_overdispersion(tmp_path):
    rng = np.random.default_rng(0)
    selection_panel = rng.gamma(0.5, size=(96, 5))
    selection_panel /= selection_panel.sum(axis=0)
    contexts = [f"ctx{i}" for i in range(selection_panel.shape[0])]
    sigs = [f"S{j}" for j in range(selection_panel.shape[1])]
    pd.DataFrame(selection_panel, index=contexts, columns=sigs).to_csv(
        tmp_path / "sigs.tsv", sep="\t"
    )
    w = np.array([[0.6, 0, 0.4, 0, 0], [0, 0.5, 0, 0.5, 0]]).T
    counts = rng.poisson(5000 * (selection_panel @ w)).astype(float)
    pd.DataFrame(counts, index=contexts, columns=["a", "b"]).to_csv(
        tmp_path / "samples.tsv", sep="\t"
    )
    main(
        [
            "--samples",
            str(tmp_path / "samples.tsv"),
            "--signatures",
            str(tmp_path / "sigs.tsv"),
            "--output",
            str(tmp_path / "out"),
            "--R",
            "20",
            "--n-jobs",
            "1",
            "--overdispersion",
            "auto",
        ]
    )
    act = read_activities(tmp_path / "out" / OUTPUT_NAME)
    assert act.shape == (2, 5)
    assert act.sum(axis=1).values == pytest.approx(counts.sum(axis=0))
