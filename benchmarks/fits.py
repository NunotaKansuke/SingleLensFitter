"""Benchmark single-band PSPL and FSPL fits on deterministic noisy data."""
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
import VBMicrolensing as vb

from singlelensfitter import FSPLFitter, PSPLFitter, auto_init


def vbm_curve(time, finite, coefficient, *, defaults=False, tolerance=1e-4):
    engine = vb.VBMicrolensing()
    engine.LoadESPLTable(str(Path(vb.__file__).parent / "data" / "ESPL.tbl"))
    if not defaults:
        engine.Tol = engine.RelTol = tolerance
        engine.minannuli = 2 if coefficient else 1
    engine.a1 = coefficient
    batch = engine.ESPLLightCurve if finite else engine.PSPLLightCurve
    epochs = time.tolist()

    def evaluate(q):
        p = [np.log(max(abs(q[2]), 1e-12)), np.log(q[1]), q[0], *q[3:]]
        return np.asarray(batch(p, epochs)[0])
    return evaluate


def fd_fit(evaluate, flux, error, seed, *, geometry=False):
    weights = error**-2
    weights /= weights.sum()
    mean = weights @ flux

    def residual(z):
        q = z.copy()
        q[0] += seed[0]
        q[1] = np.exp(z[1])
        mag = evaluate(q)
        centered = mag - weights @ mag
        weighted = weights * centered
        fs = (weighted @ (flux - mean)) / (weighted @ centered)
        return (flux - mean - fs * centered) / error

    z = seed.copy()
    z[:2] = [0., np.log(seed[1])]
    result = least_squares(residual, z, method="lm", jac="2-point", x_scale=1.,
                           max_nfev=1000, xtol=1e-6, ftol=1e-6, gtol=1e-6)
    if not result.success:
        raise RuntimeError(result.message)
    if geometry:
        q = result.x.copy()
        q[0] += seed[0]
        q[1] = np.exp(q[1])
        if len(q) == 4:
            q[3] = np.exp(q[3])
        return float(result.fun @ result.fun), q
    return float(result.fun @ result.fun)


def measure(cases, repeats=9):
    # Rotate order after warming all implementations. The cold case explicitly
    # discards the last FFT result while retaining imported code and tables.
    samples = {name: [] for name in cases}
    names = list(cases)
    for rep in range(repeats + 1):
        for name in names:
            tic = perf_counter()
            chi = cases[name]()
            elapsed = (perf_counter() - tic) * 1000
            if not np.isfinite(chi):
                raise ValueError(f"Nonfinite chi-square from {name}")
            if rep:
                samples[name].append((elapsed, chi))
        names = names[1:] + names[:1]
    return samples


def synthetic_event(finite, a):
    time = np.linspace(-12, 12, 1000)
    truth = np.array([.13, 3., .12, *([np.log(.24)] if finite else [])])
    seed = np.array([.25, 2.8, .14, *([np.log(.22)] if finite else [])])
    error = np.full_like(time, .02)
    evaluate = vbm_curve(time, finite, a)
    flux = .4 + 1.7 * evaluate(truth) + np.random.default_rng(82).normal(0, error)
    return time, flux, error, seed, evaluate


def run(label, finite, a):
    time, flux, error, seed, evaluate = synthetic_event(finite, a)
    ft = FSPLFitter(linear_limb_darkening=a) if finite else PSPLFitter()

    def slf(start):
        result = ft.fit(time, flux, error, start)
        if not result.optimizer_success:
            raise RuntimeError(result.optimizer_status)
        return float(result.chi2)

    def auto():
        auto_init._last_search = None
        return slf(None)

    samples = measure({"SLF auto": auto, "SLF seeded": lambda: slf(seed),
                       "SLF model + FD": lambda: fd_fit(lambda q: ft._magnification(time, q), flux, error, seed),
                       "VBM + FD": lambda: fd_fit(evaluate, flux, error, seed)})
    for name, rows in samples.items():
        for rep, (ms, chi) in enumerate(rows, 1):
            yield label, name, rep, ms, chi


if __name__ == "__main__":
    import sys
    writer = csv.writer(sys.stdout)
    writer.writerow(["model", "method", "repeat", "ms", "chi2"])
    for label, finite, a in [("PSPL", False, 0.), ("Uniform FSPL", True, 0.),
                             ("Linear FSPL", True, .5)]:
        writer.writerows(run(label, finite, a))
