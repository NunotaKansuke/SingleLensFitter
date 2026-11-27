"""Accuracy and runtime of seeded PSPL/FSPL methods on identical photometry."""
import csv
from contextlib import redirect_stdout
from io import StringIO
import multiprocessing
from pathlib import Path
import runpy
import sys
from time import perf_counter
from unittest.mock import patch

import MulensModel as mm
import numpy as np
from scipy.optimize import least_squares, minimize
from pyLIMA import event, telescopes
from pyLIMA.fits.TRF_fit import TRFfit
from pyLIMA.models.FSPL_model import FSPLmodel
from pyLIMA.models.FSPLarge_model import FSPLargemodel
from pyLIMA.models.PSPL_model import PSPLmodel
from singlelensfitter import FSPLFitter, PSPLFitter

from fits import fd_fit, measure, vbm_curve

quadrature = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                "tests/limb_reference.py"))["magnification"]


def reference(time, q, a, order):
    u = np.hypot((time-q[0])/q[1], q[2])
    if len(q) == 3:
        return (u*u+2)/(u*np.hypot(u, 2))
    return quadrature(u, q[3], a, order=order)


def score_reference(time, flux, err, q, a):
    mag = reference(time, q, a, 96)
    design = np.column_stack([mag, np.ones_like(mag)]) / err[:, None]
    coef = np.linalg.lstsq(design, flux/err, rcond=None)[0]
    residual = flux/err-design@coef
    return residual @ residual


def audited_fit(fn, seed):
    """Inspect the actual SciPy call on an untimed replay of the fit."""
    runs = []
    expected_start = seed.copy()
    expected_start[:2] = [0., np.log(seed[1])]
    def checked(fun, x0, **kwargs):
        np.testing.assert_array_equal(x0, expected_start)
        for key, value in dict(method="lm", x_scale=1., max_nfev=1000,
                               xtol=1e-6, ftol=1e-6, gtol=1e-6).items():
            assert kwargs[key] == value, (key, kwargs[key])
        result = least_squares(fun, x0, **kwargs)
        runs.append(result)
        return result

    with patch("fits.least_squares", checked), patch("singlelensfitter.fitters.least_squares", checked):
        chi, geometry = fn()
    assert len(runs) == 1 and runs[0].success
    return chi, geometry, [runs[0].status, runs[0].nfev]


def photometry(radius, a):
    """Generate the observations and physical starting geometry for one case."""
    finite = radius > 0
    scale = 10. if radius == .024 else 1.
    truth = np.array([.13, 3.*scale, .12/scale] + ([radius] if finite else []))
    start = np.array([.25, 2.8*scale, .14/scale] + ([radius*11/12] if finite else []))
    time = np.linspace(-12., 12., 1000)
    err = np.full_like(time, .02)
    mag = reference(time, truth, a, 96)
    assert np.max(abs(mag-reference(time, truth, a, 64))) * 1.7 < .001*err[0]
    flux = .4 + 1.7*mag + np.random.default_rng(82).normal(0., err)

    return time, flux, err, start


def run(radius, a, lee_only=False, vbm_only=False, matched=False):
    finite = radius > 0
    time, flux, err, start = photometry(radius, a)
    seed = start.copy()
    if finite:
        seed[3] = np.log(seed[3])

    fitter = FSPLFitter(linear_limb_darkening=a) if finite else PSPLFitter()

    def slf():
        options = {"numpy_result": True} if matched else {}
        r = fitter.fit(time, flux, err, seed, **options)
        if not r.optimizer_success:
            raise RuntimeError(r.optimizer_status)
        return float(r.chi2), np.asarray(r.params)

    methods = {"SLF": slf}
    for label, defaults in [("VBM batch + FD", False), ("VBM default + FD", True)]:
        if defaults and not finite:
            continue
        curve = vbm_curve(time, finite, a, defaults=defaults, tolerance=1e-3 if matched else 1e-4)
        methods[label] = lambda curve=curve: fd_fit(curve, flux, err, seed, geometry=True)

    if vbm_only:
        curve = vbm_curve(time, finite, a, tolerance=1e-3)
        def middle():
            return fd_fit(curve, flux, err, seed, geometry=True)
        label = "VBM 1e-3 + FD"
        results = measure({label: lambda: middle()[0]})
        chi, q = middle()
        score = score_reference(time, flux, err, q, a)
        for i, (ms, measured_chi) in enumerate(results[label], 1):
            np.testing.assert_allclose(measured_chi, chi)
            yield [radius, a, label, i, ms, chi, score, *q, "ok"]
        return

    phot = mm.MulensData(data_list=[time, flux, err], phot_fmt="flux", bandpass="I")
    names = ["t_0", "t_E", "u_0"] + (["rho"] if finite else [])
    model = mm.Model(dict(zip(names, start)))
    if finite:
        model.default_magnification_method = "finite_source_LD_Yoo04" if a else "finite_source_uniform_Gould94"
        model.set_limb_coeff_u("I", a)
    ev = mm.Event(datasets=[phot], model=model)
    # Scale distances relative to their initial values for the external optimizer.
    units = start.copy()
    units[0] = 1.

    def set_params(q):
        for name, val in zip(names, q):
            setattr(model.parameters, name, val)

    def mm_fit():
        def objective(x):
            set_params(x*units)
            value = ev.get_chi2()
            return value, np.asarray(ev.calculate_chi2_gradient(names))*units

        result = minimize(objective, start/units, jac=True, method="L-BFGS-B",
                          bounds=[(None, None)] + [(1e-5, None)]*(len(start)-1),
                          options={"ftol": 1e-13, "gtol": 1e-7, "maxls": 60})
        if not result.success:
            raise RuntimeError(result.message)
        return float(result.fun), result.x*units

    if not matched:
        methods["MulensModel Event + gradient"] = mm_fit

    def mm_curve(q):
        physical = q.copy()
        if finite:
            physical[3] = np.exp(q[3])
        set_params(physical)
        return model.get_magnification(time, gamma=2*a/(3-a))

    if lee_only:
        model.default_magnification_method = ("finite_source_LD_Lee09" if a
                                              else "finite_source_uniform_Lee09")
        mm_curve(seed)
        tic = perf_counter()
        chi, q = fd_fit(mm_curve, flux, err, seed, geometry=True)
        ms = (perf_counter()-tic)*1000
        yield [radius, a, "MulensModel Lee09 + FD", 1, ms, chi,
               score_reference(time, flux, err, q, a), *q, "ok"]
        return

    methods["MulensModel + FD"] = lambda: fd_fit(mm_curve, flux, err, seed, geometry=True)
    telescope = telescopes.Telescope(name="survey", lightcurve=np.column_stack((time, flux, err)),
        lightcurve_names=["time","flux","err_flux"], lightcurve_units=["JD","W/m^2","W/m^2"])
    for name, arr in zip(("time","flux","err_flux"), (time, flux, err)):
        np.testing.assert_array_equal(telescope.lightcurve[name].value, arr)
    telescope.ld_a1, telescope.ld_gamma = a, 2*a/(3-a)
    data = event.Event()
    data.telescopes.append(telescope)
    choices = [("pyLIMA PSPL", PSPLmodel)] if not finite else [
        ("pyLIMA FSPL", FSPLmodel), ("pyLIMA FSPLarge", FSPLargemodel)]
    if matched:
        # Control for the derivative implementation with the same FD driver.
        methods["SLF model + FD"] = lambda: fd_fit(
            lambda q: fitter._magnification(time, q), flux, err, seed, geometry=True)
    for label, cls in choices:
        if matched:
            lens = cls(data)
            def curve(q, lens=lens):
                pars = {"t0": q[0], "tE": q[1], "u0": q[2]}
                if finite:
                    pars["rho"] = np.exp(q[3])
                return lens.model_magnification(telescope, pars)
            methods[label + " + FD"] = lambda curve=curve: fd_fit(
                curve, flux, err, seed, geometry=True)
            continue
        trf = TRFfit(cls(data))
        if finite and radius > .05:
            trf.fit_parameters["rho"][1] = (5e-5, 1.)

        def fit(trf=trf):
            trf.model_parameters_guess = [start[0], start[2], start[1], *start[3:]]
            trf.telescopes_fluxes_parameters_guess = []
            trf.fit_results = {}
            with redirect_stdout(StringIO()):
                trf.fit()
            r = trf.fit_results
            if not r or not r["fit_object"].success:
                raise RuntimeError("pyLIMA fit failed")
            q = r["best_model"][[0, 2, 1] + ([3] if finite else [])]
            return float(r["chi2"]), q
        methods[label] = fit

    # Check each workflow once before timing. Preserve optimizer failures as rows.
    for label in list(methods):
        try:
            methods[label]()
        except RuntimeError as failure:
            yield [radius, a, label, 0, "", "", "", "", "", "", "", str(failure), *(["", ""] if matched else [])]
            del methods[label]
    samples = measure({k: lambda f=f: f()[0] for k, f in methods.items()})
    for label, fn in methods.items():
        audit = []
        if matched:
            chi, q, audit = audited_fit(fn, seed)
        else:
            chi, q = fn()
        reference_chi = score_reference(time, flux, err, q, a)
        for i, (ms, measured_chi) in enumerate(samples[label], 1):
            np.testing.assert_allclose(measured_chi, chi, rtol=1e-9)
            yield [radius, a, label, i, ms, chi, reference_chi, *q[:3], q[3] if finite else "", "ok", *audit]


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--matched", action="store_true", help="common LM seed, scaling and stopping rules")
    modes.add_argument("--lee09", action="store_true", help="single large-source integration fits")
    modes.add_argument("--vbm-sweep", action="store_true", help="intermediate 1e-3 VBM tolerance")
    args = parser.parse_args()
    writer.writerow(["rho_true","a","method","repeat","ms","chi2","reference_chi2","t0","tE","u0","rho_fit","status"] + (["lm_status", "nfev"] if args.matched else []))
    lee = args.lee09
    cases = [(.24,0), (.24,.5)] if lee else [(0,0),(.024,0),(.024,.5),(.24,0),(.24,.5)]
    for radius, a in cases:
        if args.vbm_sweep and radius == 0:
            continue
        writer.writerows(run(radius, a, lee, args.vbm_sweep, args.matched))
        sys.stdout.flush()
