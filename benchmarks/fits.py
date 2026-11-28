"""Shared helpers for single-band PSPL and FSPL benchmarks."""
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
import VBMicrolensing as vb


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
