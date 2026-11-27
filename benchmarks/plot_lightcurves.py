"""Render the synthetic photometry and the saved SLF fits from the matched benchmark."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from methods import FSPLFitter, PSPLFitter, photometry

root = Path(__file__).resolve().parent


def plot_case(row, output):
    radius, a = float(row["rho_true"]), float(row["a"])
    time, flux, err, _ = photometry(radius, a)
    params = [float(row[name]) for name in ("t0", "tE", "u0")]
    if radius:
        params.append(np.log(float(row["rho_fit"])))
    fitter = FSPLFitter(linear_limb_darkening=a) if radius else PSPLFitter()
    fit = fitter.evaluate_fixed(time, flux, err, params)
    residual = np.asarray(fit.residual) / err
    np.testing.assert_allclose(residual @ residual, float(row["chi2"]), rtol=1e-9)

    grid = np.linspace(time.min(), time.max(), 10001)
    model = float(fit.fs) * fitter._magnification(grid, np.asarray(params)) + float(fit.fb)
    fig, axes = plt.subplots(2, 2, figsize=(9, 4.3), sharex="col",
                             height_ratios=[3, 1], layout="constrained")
    for (curve, residuals), limits in zip(axes.T, [(-12, 12), (-1.4, 1.6)]):
        observed = (time >= limits[0]) & (time <= limits[1])
        smooth = (grid >= limits[0]) & (grid <= limits[1])
        curve.errorbar(time[observed], flux[observed], yerr=err[observed],
                       fmt=".", ms=3, elinewidth=.6, alpha=.65)
        curve.plot(grid[smooth], model[smooth], lw=1.5)
        residuals.plot(time[observed], residual[observed], ".", ms=2.6, alpha=.65)
        residuals.set(xlim=limits, ylim=(-4.5, 4.5), xlabel="Time (days)")
    assert np.max(np.abs(residual)) < 4.5
    axes[0, 0].set_ylabel("Flux (arbitrary units)")
    axes[1, 0].set_ylabel(r"Residual ($\sigma$)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    with (root / "matched_fits.csv").open() as table:
        for row in csv.DictReader(table):
            if row["method"] == "SLF" and row["repeat"] == "1":
                profile = "linear" if float(row["a"]) else "uniform"
                name = f"rho-{row['rho_true']}-{profile}" if float(row["rho_true"]) else "pspl"
                plot_case(row, directory / f"{name}.png")
