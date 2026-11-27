"""Repeated single-band fits and physical-coordinate objectives on fixed epochs."""
from copy import copy
from dataclasses import replace

import numpy as np

from .fitters import FSPLFitter, _cpp_fspl_magnification_gradient, _fit_arrays
from .parallax_backend import _ParallaxFitter
from .singlelens_fit import SingleLensFitResult


def expand_masked_fit(fitter, subset, data, excluded):
    """Restore predictions at excluded epochs without refitting linear fluxes."""
    time, flux, ferr = data
    keep = ~excluded
    model = np.empty_like(time)
    model[keep] = subset.model_flux
    model[excluded] = float(subset.fs) * fitter._magnification(
        time[excluded], fitter.seed_from_fit(subset)) + float(subset.fb)
    unfinished = subset.unfinished_candidate
    return replace(subset, time=time, flux=flux, ferr=ferr, model_flux=model,
                   residual=flux-model, mask=excluded.copy(),
                   unfinished_candidate=expand_masked_fit(fitter, unfinished, data, excluded) if unfinished is not None else None)


class PreparedCurve:
    """A snapshot of one light curve and fitter configuration; use one per worker."""

    def __init__(self, fitter, time, flux, ferr):
        if any(np.ndim(x) != 1 for x in (time, flux, ferr)):
            raise ValueError("time, flux, and ferr must be one-dimensional")
        self._data = tuple(x.copy() for x in _fit_arrays(time, flux, ferr))
        for x in self._data:
            x.flags.writeable = False
        if self._data[0].size == 0:
            raise ValueError("a prepared curve requires observations")
        self._fitter = copy(fitter)
        self._parallax = isinstance(fitter, _ParallaxFitter)
        self._finite = fitter.finite_source if self._parallax else isinstance(fitter, FSPLFitter)
        self._evaluator = fitter._evaluator(*self._data) if self._parallax else None
        self._weights = 1.0 / self._data[2]**2
        self._params = None

    def _excluded(self, mask):
        if mask is None:
            return None
        excluded = np.asarray(mask)
        if excluded.dtype != bool or excluded.shape != self._data[0].shape:
            raise ValueError("mask must be boolean and have the same shape as time")
        return excluded if excluded.any() else None

    def _params_array(self, params):
        q = np.asarray(params, dtype=float)
        if q.shape != (self._fitter.parameter_dimension,) or not np.isfinite(q).all() or q[1] <= 0:
            raise ValueError("expected finite physical fit parameters with positive tE")
        if self._finite and not 1e-12 <= q[3] <= 100:
            raise ValueError("rho must be between 1e-12 and 100")
        if getattr(self._fitter, "fit_limb_darkening", False) and not 0 <= q[4] <= 1:
            raise ValueError("limb-darkening coefficient must be in [0, 1]")
        return q

    def _partials(self, q):
        if not np.array_equal(q, self._params):
            tau = (self._data[0] - q[0]) / q[1]
            u = np.hypot(tau, q[2])
            if self._finite:
                fitter = self._fitter
                a = q[4] if fitter.fit_limb_darkening else fitter.linear_limb_darkening
                mag, du, *extra = _cpp_fspl_magnification_gradient(
                    u, q[3], espl_table_path=fitter.espl_table_path,
                    tol=fitter.magnification_tol, reltol=fitter.magnification_reltol,
                    linear_limb_darkening=a, coefficient_gradient=fitter.fit_limb_darkening,
                )
                slope = np.divide(du, u, out=np.zeros_like(u), where=u > 0)
            else:
                at_floor = u < 1e-12
                u = np.maximum(u, 1e-12)
                root = np.hypot(u, 2)
                mag = (u*u + 2) / (u * root)
                slope = -8 / ((u * root)**3)
                slope[at_floor] = 0.
                extra = []
            along = -slope * tau / q[1]
            self._derivatives = np.stack((along, along*tau, slope*q[2], *extra))
            self._mag = mag
            self._params = q.copy()
        return self._mag, self._derivatives

    def chi2_and_grad(self, params, *, mask=None):
        """Return profiled chi-square and its gradient in result.params order."""
        q = self._params_array(params)
        excluded = self._excluded(mask)
        if self._parallax:
            raw = q.copy()
            raw[0] += self._fitter.time_spec.offset
            raw[1] = np.log(q[1])
            if self._finite:
                raw[3] = np.log(q[3])
            if excluded is not None and excluded.all():
                raise ValueError("mask excludes all observations")
            chi2, grad = self._evaluator._native.chi2_and_grad(
                raw, None if excluded is None else ~excluded)
            grad[1] /= q[1]
            if self._finite:
                grad[3] /= q[3]
            return chi2, grad

        mag, derivatives = self._partials(q)
        w = self._weights if excluded is None else np.where(excluded, 0., self._weights)
        total = w.sum()
        if total == 0:
            raise ValueError("mask excludes all observations")
        flux = self._data[1]
        mean_flux = np.dot(w, flux) / total
        centered = mag - np.dot(w, mag) / total
        wx = w * centered
        xx = np.dot(wx, centered)
        if not np.isfinite(xx) or xx <= 0:
            raise ValueError("source and blend fluxes are degenerate")
        delta = flux - mean_flux
        fs = np.dot(wx, delta) / xx
        residual = delta - fs * centered
        wr = w * residual
        # The derivatives of the profiled fluxes cancel by the normal equations.
        return np.dot(wr, residual), (-2*fs) * np.einsum("ij,j->i", derivatives, wr)

    def fit(self, *, mask=None, start=None):
        """Refit kept observations and return predictions at every prepared epoch."""
        fitter = self._fitter
        excluded = self._excluded(mask)
        if isinstance(start, SingleLensFitResult):
            family = "fspl" if self._finite else "pspl"
            if self._parallax:
                family += "_parallax"
            if start.model_kind != family:
                raise ValueError("start must be a result from the same model family")
            start = start.params
        seed = None
        if start is not None:
            seed = self._params_array(start).copy()
            if self._finite:
                seed[3] = np.log(seed[3])
        keep = None if excluded is None else ~excluded
        if keep is not None and np.count_nonzero(keep) < fitter.parameter_dimension + 3:
            raise ValueError("mask leaves too few observations for this fit")
        if self._parallax:
            return fitter._fit_prepared(self._evaluator, *self._data, seed, keep=keep)
        time, flux, ferr = self._data
        selected = self._data if keep is None else (time[keep], flux[keep], ferr[keep])
        if seed is None:
            result = fitter.fit(*selected)
        else:
            result = fitter._fit_from_seed(*selected, seed, numpy_result=True)
        if keep is None:
            return result

        return expand_masked_fit(fitter, result, self._data, excluded)
