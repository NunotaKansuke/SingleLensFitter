"""Plot median full-fit times for the methods shown in comparison.md."""
import csv
from itertools import chain
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = [
    (0, 0, "Point source", "PSPL"),
    (.024, 0, "ρ = 0.024, uniform", "FSPL"),
    (.024, .5, "ρ = 0.024, linear", "FSPL"),
    (.24, 0, "ρ = 0.24, uniform", "FSPLarge"),
    (.24, .5, "ρ = 0.24, linear", "FSPLarge"),
]


def main():
    samples = defaultdict(list)
    with (ROOT / "matched_fits.csv").open() as stream, (ROOT / "singlens_fits.csv").open() as extra, (ROOT / "package_workflows.csv").open() as workflows:
        for row in chain(csv.DictReader(stream), csv.DictReader(extra), (row for row in csv.DictReader(workflows) if row["method"].startswith("pyLIMA"))):
            if row["status"] == "ok" and (
                not row["method"].startswith("pyLIMA") or not row["method"].endswith(" + FD")
            ):
                key = float(row["rho_true"]), float(row["a"]), row["method"]
                samples[key].append(float(row["ms"]))

    fig, ax = plt.subplots(figsize=(8, 5.8), layout="constrained")
    positions = np.arange(len(CASES))
    labels = ["SingleLensFitter", "VBM + finite-difference LM", "MulensModel", "pyLIMA TRFfit", "singlens"]
    for index, label in enumerate(labels):
        times = []
        for rho, a, _, pylima_model in CASES:
            method = ["SLF", "VBM batch + FD", "MulensModel + FD",
                      f"pyLIMA {pylima_model}", "singlens autodiff"][index]
            values = samples[rho, a, method]
            if len(values) != 9:
                raise ValueError(f"Expected nine timings for {rho}, {a}, {method}")
            times.append(median(values))
        centers = positions + (index - 2) * .16
        ax.barh(centers, times, height=.13)
        ax.annotate(label, (times[0], centers[0]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(positions, [case[2] for case in CASES])
    ax.invert_yaxis()
    ax.set_xlabel("Median fit time (ms)")
    ax.margins(y=.025)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", which="both", left=False, right=False, pad=6)
    ax.tick_params(axis="x", which="both", top=False, direction="out")
    fig.savefig(ROOT / "figures" / "matched-fit-times.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
