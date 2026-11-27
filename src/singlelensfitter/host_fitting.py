"""Single-lens fitting policy for the jacscanomaly host search.

Keep its ordered starts, single pass per start and minimum-chi-square selection.
The seeded optimizers, evaluators and template banks are shared with the
standalone fitters.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from . import fitters as core
from . import parallax_backend as px
from .auto_init import (
    SEED_MAX,
    SEED_TE_FACTORS,
    SEED_U0_FACTORS,
    fspl_initial_guesses,
    pspl_initial_guesses,
    seed_family,
)
from .fitters import _fit_arrays
from .init_config import InitConfig
from .prepared import expand_masked_fit
from .singlelens_fit import SingleLensFitResult

NESTED_RHO = 1.0e-4


def automatic_seeds(
    time,
    flux,
    ferr,
    *,
    finite_source: bool,
    parallax: bool = False,
    max_seeds: int = SEED_MAX,
    pspl_params=None,
    observed_scale=None,
    magnification_tol: float = 1.0e-4,
    magnification_reltol: float = 1.0e-4,
    espl_table_path: Optional[str] = None,
    config: Optional[InitConfig] = None,
) -> tuple[np.ndarray, ...]:
    """The ordered multistart set used by the host-search fitters.

    Anchors are the data-driven initial guesses (PSPL FFT bank, or the FSPL
    mixed FFT bank, with an observed-width fallback), followed by
    (with ``parallax=True`` only) the ``tE x u0`` degeneracy family -- each with
    both ``u0`` signs -- around each anchor in priority order, capped at
    ``max_seeds``.  Assumptions:
    ``rho`` is not re-gridded around finite-source anchors, and parallax
    starts at ``piE = 0`` (no direction hint exists without a detector).
    ``pspl_params`` contributes only its ``t0``, as a rough locator of the
    brightening to fit (its ``tE`` and ``u0`` are never used).
    """
    if finite_source:
        anchors = fspl_initial_guesses(
            time,
            flux,
            ferr,
            top_k=(config or InitConfig()).auto_init_fspl_template_top_k,
            config=config,
            pspl_params=pspl_params,
            observed_scale=observed_scale,
            magnification_tol=magnification_tol,
            magnification_reltol=magnification_reltol,
            espl_table_path=espl_table_path,
            parallax=parallax,
        )
    else:
        t0_hint = None
        if (
            pspl_params is not None
            and np.size(pspl_params) >= 1
            and np.isfinite(pspl_params[0])
        ):
            t0_hint = float(pspl_params[0])
        anchors = pspl_initial_guesses(
            time, flux, ferr, config=config, parallax=parallax, t0_hint=t0_hint
        )
    seeds = list(anchors)
    # Without parallax the model depends on u0 only through |u|, so the sign
    # flip is an exact duplicate and the tE x u0 family was never part of the
    # point-lens fits: keep just the anchors.  With parallax the sign selects
    # physically different solutions, so the family is added.
    for anchor in anchors if parallax else ():
        seeds.extend(
            seed_family(
                anchor,
                tE_factors=SEED_TE_FACTORS,
                u0_factors=SEED_U0_FACTORS,
                u0_sign_flip=True,
            )
        )
    unique: list[np.ndarray] = []
    for seed in seeds:
        if not any(
            np.allclose(seed, other, rtol=0.0, atol=1.0e-12) for other in unique
        ):
            unique.append(np.asarray(seed, dtype=float))
        if len(unique) >= max(1, int(max_seeds)):
            break
    return tuple(unique)


def best_fit_over_seeds(fit_one, seeds, label: str):
    """Fit every seed with ``fit_one(seed)`` and return the lowest-chi-square fit."""
    best, best_chi2, errors = None, np.inf, []
    for seed in seeds:
        try:
            candidate = fit_one(seed)
            chi2 = float(np.asarray(candidate.chi2))
        except Exception as exc:
            errors.append(exc)
            continue
        if np.isfinite(chi2) and chi2 < best_chi2:
            best, best_chi2 = candidate, chi2
    if best is None:
        message = f"All automatic {label} initial guesses failed."
        if errors:
            message += f" First error: {errors[0]}"
        raise RuntimeError(message)
    return best


def _mask(mask, size, dimension):
    if mask is None:
        return None
    excluded = np.asarray(mask, dtype=bool).reshape(-1)
    if excluded.size != size:
        raise ValueError("mask must have the same length as time")
    if not excluded.any():
        return None
    if np.count_nonzero(~excluded) < dimension + 3:
        raise ValueError("mask leaves too few points to fit the model")
    return excluded


class _HostFit:
    def fit(
        self,
        time,
        flux,
        ferr,
        p0=None,
        *,
        mask=None,
        pspl_params=None,
        observed_scale=None,
    ):
        data = _fit_arrays(time, flux, ferr)
        excluded = _mask(mask, data[0].size, self.parameter_dimension)
        selected = data if excluded is None else tuple(x[~excluded] for x in data)
        if p0 is None and self.model_kind == "fspl":
            fit = super().fit(*selected, observed_scale=observed_scale, pspl_params=pspl_params)
        elif p0 is None:
            seeds = self._automatic_initial_guesses(
                *selected, pspl_params=pspl_params, observed_scale=observed_scale
            )
            fit = best_fit_over_seeds(
                lambda p: self._fit_from_seed(*selected, p), seeds, type(self).__name__
            )
        else:
            fit = self._fit_from_seed(*selected, p0)
        if excluded is not None:
            fit = expand_masked_fit(self, fit, data, excluded)
        self._last_fit = fit
        return fit

    def evaluate_fixed(self, time, flux, ferr, p0, *, mask=None):
        if mask is None:
            return super().evaluate_fixed(time, flux, ferr, p0)
        data = _fit_arrays(time, flux, ferr)
        excluded = _mask(mask, data[0].size, self.parameter_dimension)
        if excluded is None:
            return super().evaluate_fixed(*data, p0)
        fit = super().evaluate_fixed(*(x[~excluded] for x in data), p0)
        self._last_fit = expand_masked_fit(self, fit, data, excluded)
        return self._last_fit

    def fit_fixed_model(self, time, flux, ferr, p0, *, model_kind=None, mask=None):
        if model_kind not in (None, self.model_kind):
            raise ValueError(f"Expected model_kind={self.model_kind!r}")
        return self.evaluate_fixed(time, flux, ferr, p0, mask=mask)

    def magnification_model(self, time):
        epochs = np.asarray(time, dtype=float)
        return lambda seed: self._magnification(epochs, np.asarray(seed, dtype=float))


class PSPLFitter(_HostFit, core.PSPLFitter):
    model_kind = "pspl"

    def _automatic_initial_guesses(self, *data, pspl_params=None, observed_scale=None):
        return automatic_seeds(
            *data, finite_source=False, config=self.init_config, pspl_params=pspl_params
        )

    @staticmethod
    def nested_seed(pspl_params) -> np.ndarray:
        """The point of this model that is the PSPL ``(t0, tE, u0)``: itself."""
        return np.asarray(pspl_params, dtype=float).reshape(-1)[:3].copy()

    def fit_robust(
        self,
        time,
        flux,
        ferr,
        p0,
        *,
        t0_bounds,
        tE_bounds=(0.05, 1000.0),
        u0_bounds=(1.0e-5, 3.0),
        loss="cauchy",
        f_scale=2.5,
    ) -> SingleLensFitResult:
        """Fit one identified event robustly using every supplied observation.

        Only the peak time is restricted to ``t0_bounds``; observations outside
        that interval remain in the objective. Source flux is positive and
        blend flux is unrestricted. Unlike ``fit``, source and blend are fitted
        jointly under the requested robust loss, not profiled by least squares.
        ``Finder.scan_fit`` preserves this complete fitted model. In contrast,
        ``Finder.run(refit=False)`` solves new least-squares linear fluxes.
        """
        t, f, e = _fit_arrays(time, flux, ferr)
        if (
            loss not in {"cauchy", "soft_l1"}
            or not np.isfinite(f_scale)
            or f_scale <= 0
        ):
            raise ValueError(
                "A supported robust loss and positive f_scale are required."
            )
        bounds = [np.asarray(v, dtype=float) for v in (t0_bounds, tE_bounds, u0_bounds)]
        if any(
            v.shape != (2,) or not np.isfinite(v).all() or v[0] >= v[1] for v in bounds
        ):
            raise ValueError("Parameter bounds must be finite increasing pairs.")
        if bounds[1][0] <= 0 or bounds[2][0] <= 0:
            raise ValueError("tE and u0 bounds must be positive.")
        seed = np.asarray(p0, dtype=float)
        if seed.shape != (3,) or not np.isfinite(seed).all():
            raise ValueError("p0 must contain finite (t0, tE, u0).")
        origin = float(np.mean(bounds[0]))
        dt = t - origin
        scale = float(np.median(e))
        lower = np.array(
            [
                bounds[0][0] - origin,
                np.log(bounds[1][0]),
                np.log(bounds[2][0]),
                0.0,
                -np.inf,
            ]
        )
        upper = np.array(
            [
                bounds[0][1] - origin,
                np.log(bounds[1][1]),
                np.log(bounds[2][1]),
                np.inf,
                np.inf,
            ]
        )
        geometry = np.array(
            [
                seed[0] - origin,
                np.log(max(abs(seed[1]), 1e-12)),
                np.log(max(abs(seed[2]), 1e-12)),
            ]
        )
        geometry = np.clip(geometry, lower[:3] + 1e-8, upper[:3] - 1e-8)

        def profile(z):
            q = np.array([z[0], np.exp(z[1]), np.exp(z[2])])
            peak = float(self._magnification(np.array([q[0]]), q)[0] - 1.0)
            return (self._magnification(dt, q) - 1.0) / peak

        baseline = float(np.median(f)) / scale
        shape = profile(geometry)
        near = shape > 0.25
        amplitude = (
            max(float(np.percentile(f[near], 75)) / scale - baseline, 1.0)
            if near.any()
            else 1.0
        )
        initial = np.r_[geometry, amplitude, baseline]
        result = least_squares(
            lambda z: (scale * (z[4] + z[3] * profile(z)) - f) / e,
            initial,
            bounds=(lower, upper),
            loss=loss,
            f_scale=f_scale,
            x_scale="jac",
            max_nfev=self.maxiter,
            ftol=self.tol,
            xtol=self.tol,
            gtol=self.tol,
        )
        z = result.x
        q = np.array([z[0] + origin, np.exp(z[1]), np.exp(z[2])])
        peak = float(self._magnification(np.array([q[0]]), q)[0] - 1.0)
        fs = scale * z[3] / peak
        fb = scale * z[4] - fs
        model = fs * self._magnification(t, q) + fb
        residual = f - model
        chi2 = float(np.sum((residual / e) ** 2))
        fit = SingleLensFitResult(
            time=t,
            flux=f,
            ferr=e,
            params=jnp.asarray(q),
            param_names=("t0", "tE", "u0"),
            fs=jnp.asarray(fs),
            fb=jnp.asarray(fb),
            model_flux=jnp.asarray(model),
            residual=jnp.asarray(residual),
            chi2=jnp.asarray(chi2),
            chi2_dof=jnp.asarray(chi2 / max(len(t) - 5, 1)),
            model_kind="pspl",
            optimizer_success=bool(result.success),
            optimizer_status=f"scipy_trf_robust:{result.status}:{result.message}",
            diagnostics=dict(
                loss=loss,
                f_scale=float(f_scale),
                cost=float(result.cost),
                nfev=int(result.nfev),
                t0_bounds=bounds[0].tolist(),
                points=len(t),
            ),
        )
        self._last_fit = fit
        return fit


class FSPLFitter(_HostFit, core.FSPLFitter):
    model_kind = "fspl"

    @staticmethod
    def nested_seed(pspl_params) -> np.ndarray:
        """The FSPL start that is the PSPL ``(t0, tE, u0)``: ``rho -> 0``.

        ``rho = NESTED_RHO`` is far below any ``u0`` a host fit resolves, so
        the start reproduces the PSPL magnification and its chi2.
        """
        p = np.asarray(pspl_params, dtype=float).reshape(-1)
        return np.asarray([p[0], p[1], p[2], np.log(NESTED_RHO)], dtype=float)

    def _automatic_initial_guesses(self, *data, pspl_params=None, observed_scale=None):
        seeds = super()._automatic_initial_guesses(*data, observed_scale=observed_scale)
        if pspl_params is not None:
            return (*seeds, self.nested_seed(pspl_params))
        return seeds


class _HostParallax:
    def fit(
        self,
        time,
        flux,
        ferr,
        q0=None,
        *,
        mask=None,
        pspl_params=None,
        observed_scale=None,
    ):
        data = _fit_arrays(time, flux, ferr)
        excluded = _mask(mask, data[0].size, self.parameter_dimension)
        return self._fit_prepared(
            self._evaluator(*data),
            *data,
            q0,
            pspl_params,
            observed_scale,
            None if excluded is None else ~excluded,
        )

    def _fit_prepared(
        self,
        evaluator,
        time,
        flux,
        ferr,
        q0,
        pspl_params=None,
        observed_scale=None,
        keep=None,
    ):
        def fit_seed(seed):
            return self._fit_seed(evaluator, time, flux, ferr, seed, keep=keep)

        if q0 is not None:
            return fit_seed(q0)
        data = (time, flux, ferr)
        selected = data if keep is None else tuple(x[keep] for x in data)
        seeds = self._automatic_fspl_initial_guesses(
            *selected, pspl_params=pspl_params, observed_scale=observed_scale
        )
        self._last_fit = best_fit_over_seeds(fit_seed, seeds, "parallax")
        return self._last_fit

    def magnification_model(self, time):
        epochs = np.asarray(time, dtype=float)
        ones = np.ones_like(epochs)
        evaluator = self._evaluator(epochs, ones, ones)
        return lambda seed: evaluator.magnification(self._raw_seed(seed))

    def nested_seed(self, pspl_params) -> np.ndarray:
        """The start that is the PSPL ``(t0, tE, u0)``: ``piE = 0`` (and ``rho -> 0``)."""
        p = np.asarray(pspl_params, dtype=float).reshape(-1)[:3]
        rho = [np.log(NESTED_RHO)] if self.finite_source else []
        return np.r_[p, rho, 0.0, 0.0]

    def _automatic_fspl_initial_guesses(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        ferr: np.ndarray,
        *,
        pspl_params=None,
        observed_scale=None,
    ) -> tuple[np.ndarray, ...]:
        """Shared multistart set, with parallax starting at zero.

        With ``pspl_params`` the PSPL solution itself (:meth:`nested_seed`) is
        one more start, so the fit cannot end worse than the PSPL it nests.
        """

        seeds = super()._automatic_fspl_initial_guesses(
            time, flux, ferr, pspl_params=pspl_params, observed_scale=observed_scale
        ) if self.finite_source else automatic_seeds(
            time, flux, ferr, finite_source=False, parallax=True,
            pspl_params=pspl_params, config=self.init_config,
        )
        if pspl_params is None:
            return seeds
        return (*seeds, self.nested_seed(pspl_params))


class PSPLParallaxFitter(_HostParallax, px.PSPLParallaxFitter):
    pass


class FSPLParallaxFitter(_HostParallax, px.FSPLParallaxFitter):
    pass
