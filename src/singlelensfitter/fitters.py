"""Canonical single-lens fitters.

There are only four model families in the public fitting API:

* :class:`PSPLFitter`
* :class:`FSPLFitter`
* :class:`PSPLParallaxFitter`
* :class:`FSPLParallaxFitter`

The model family owns the fit contract.  Observer geometry (annual or space)
and the space-observer convention (including GULLS) are options of the
parallax fitters, not separate fitter classes.  Continuous optimization is
performed by SciPy's Levenberg--Marquardt solver.  PSPL uses the analytic
point-source kernel; FSPL and parallax use compiled magnification/trajectory
evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import jax.numpy as jnp

from .auto_init import automatic_seeds, best_fit_over_seeds
from .init_config import InitConfig
from .singlelens_fit import SingleLensFitResult

from .parallax_backend import (
    FSPLParallaxFitter,
    PSPLParallaxFitter,
    default_espl_table_path,
)


class _LazyPlotter:
    """Plotting is not ported yet; it is loaded from jacscanomaly on first use."""

    def __getattr__(self, name):
        try:
            from jacscanomaly.plot import SingleLensPlotter
        except ImportError as exc:  # pragma: no cover
            raise ImportError("plotting needs jacscanomaly.plot") from exc
        return getattr(SingleLensPlotter(), name)


def _make_plotter():
    return _LazyPlotter()

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - dependency is declared by the package
    least_squares = None

try:
    from . import _vbm_cpp
except ImportError:  # pragma: no cover - source-only installations
    _vbm_cpp = None

try:
    import VBMicrolensing
except ImportError:  # pragma: no cover - dependency is declared by the package
    VBMicrolensing = None


class _FluxProfile:
    """Weighted two-column regression with fixed observations during a fit."""

    def __init__(self, flux, ferr):
        self.errors = np.asarray(ferr, dtype=float)
        self.weights = 1.0 / (self.errors * self.errors)
        self.weight_sum = np.sum(self.weights)
        self.y_mean = np.sum(self.weights * flux) / self.weight_sum
        self.weighted_y = self.weights * (flux - self.y_mean)
        self.magnification = None

    def solve(self, magnification):
        if self.magnification is not magnification:
            mean = np.sum(self.weights * magnification) / self.weight_sum
            self.centered = magnification - mean
            self.weighted_x = self.weights * self.centered
            self.denominator = np.sum(self.weighted_x * self.centered)
            self.fs = (np.sum(self.weighted_y * self.centered) / self.denominator
                       if self.denominator > 0 and np.isfinite(self.denominator) else np.nan)
            self.fb = self.y_mean - self.fs * mean
            self.magnification = magnification
        return self.fs, self.fb

    def jacobian(self, magnification, gradient):
        self.solve(magnification)
        projection = self.weighted_y - 2 * self.fs * self.weighted_x
        dfs = (projection @ gradient) / self.denominator
        dmean = (self.weights @ gradient) / self.weight_sum
        for j, column in enumerate(gradient.T):
            column -= dmean[j]
            column *= -self.fs
            column -= self.centered * dfs[j]
            column /= self.errors
        return gradient


def _solve_fs_fb_numpy(magnification, flux, ferr):
    return _FluxProfile(flux, ferr).solve(magnification)


def _fit_arrays(time, flux, ferr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_np = np.asarray(time, dtype=float).reshape(-1)
    flux_np = np.asarray(flux, dtype=float).reshape(-1)
    ferr_np = np.asarray(ferr, dtype=float).reshape(-1)
    if not (time_np.shape == flux_np.shape == ferr_np.shape):
        raise ValueError("time, flux, and ferr must have equal one-dimensional shapes.")
    if not np.all(np.isfinite(time_np)) or not np.all(np.isfinite(flux_np)):
        raise ValueError("time and flux must be finite.")
    if not np.all(np.isfinite(ferr_np)) or np.any(ferr_np <= 0.0):
        raise ValueError("ferr must be finite and positive.")
    return time_np, flux_np, ferr_np


def _pspl_magnification(time: np.ndarray, t0: float, tE: float, u0: float) -> np.ndarray:
    tE_safe = max(abs(float(tE)), 1.0e-12)
    tau = (np.asarray(time, dtype=float) - float(t0)) / tE_safe
    u = np.sqrt(tau * tau + float(u0) * float(u0))
    u_safe = np.maximum(u, 1.0e-12)
    return (u_safe * u_safe + 2.0) / (
        u_safe * np.sqrt(u_safe * u_safe + 4.0)
    )


def _profiled_residual_jacobian(
    magnification: np.ndarray,
    magnification_gradient: np.ndarray,
    flux: np.ndarray,
    ferr: np.ndarray,
    *,
    overwrite_gradient: bool = False,
) -> np.ndarray:
    """Jacobian of the profiled-flux residual (variable projection).

    ``magnification_gradient`` is dA/dz with shape (n, k).  With
    r = (y - ym - fs (A - xm)) / sigma and fs = Sxy / Sxx the exact derivative
    including the dependence of the profiled fs and fb on the parameters is
    dr = -(dfs (A - xm) + fs (dA - dxm)) / sigma.
    """
    gradient = (magnification_gradient if overwrite_gradient else
                np.array(magnification_gradient, dtype=float, order="K", copy=True))
    return _FluxProfile(flux, ferr).jacobian(magnification, gradient)


def _trajectory_gradient(
    time: np.ndarray, t0: float, tE: float, u0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return u(t) and du/d(t0, ln tE, u0) for the rectilinear trajectory."""
    tE_safe = max(abs(float(tE)), 1.0e-12)
    tau = (np.asarray(time, dtype=float) - float(t0)) / tE_safe
    u = np.sqrt(tau * tau + float(u0) * float(u0))
    u_safe = np.maximum(u, 1.0e-12)
    du = np.column_stack(
        [-tau / (tE_safe * u_safe), -tau * tau / u_safe, np.full_like(u, float(u0)) / u_safe]
    )
    return u, du


def _pspl_gradient(time: np.ndarray, t0: float, tE: float, u0: float) -> np.ndarray:
    """dA/d(t0, ln tE, u0) of the point-source point-lens magnification."""
    u, du = _trajectory_gradient(time, t0, tE, u0)
    u_safe = np.maximum(u, 1.0e-12)
    u2 = u_safe * u_safe
    dA_du = -8.0 / (u2 * (u2 + 4.0) * np.sqrt(u2 + 4.0))
    return dA_du[:, None] * du


def _run_scipy_lm(
    residual,
    initial: np.ndarray,
    *,
    maxiter: int,
    tol: float,
    diff_step: Optional[float] = None,
    jac="2-point",
):
    if least_squares is None:
        raise ImportError("scipy is required for single-lens fitting.")
    tolerance = max(float(tol), 1.0e-14)
    kwargs = {
        "jac": jac,
        "method": "lm",
        "max_nfev": max(1, int(maxiter)),
        "xtol": tolerance,
        "ftol": tolerance,
        "gtol": tolerance,
    }
    if diff_step is not None and jac == "2-point":
        kwargs["diff_step"] = float(diff_step)
    return least_squares(residual, np.asarray(initial, dtype=float), **kwargs)


def _optimizer_status(result, *, maxiter: int, detail: str = "", method: str = "lm") -> str:
    if bool(getattr(result, "fixed", False)):
        return "fixed_parameters"
    suffix = f";{detail}" if detail else ""
    return (
        f"scipy_{method}:status={int(result.status)};"
        f"nfev={int(result.nfev)};max_nfev={int(maxiter)};"
        f"message={result.message}{suffix}"
    )


_LOG_MAX_ESPL_RADIUS = float(np.log(100.0))


def _espl_radius(logrho: float) -> float:
    # Map the last table node exactly, including exp/log roundoff.
    if logrho == _LOG_MAX_ESPL_RADIUS:
        return 100.0
    return float(np.exp(min(max(logrho, -50.0), 10.0)))


def _cpp_fspl_magnification(
    u: np.ndarray,
    rho: float,
    *,
    espl_table_path: Optional[str],
    tol: float,
    reltol: float,
    linear_limb_darkening: float = 0.,
) -> np.ndarray:
    """Evaluate FSPL magnification in the compiled VBM-backed extension."""
    u_np = np.asarray(u, dtype=float)
    rho_safe = max(float(rho), 1.0e-12)
    if not rho_safe <= 100.0:
        raise ValueError("ESPL requires rho <= 100")
    if _vbm_cpp is not None and hasattr(_vbm_cpp, "fspl_magnification"):
        values = _vbm_cpp.fspl_magnification(
            u_np,
            rho_safe,
            espl_table=espl_table_path,
            tol=float(tol),
            reltol=float(reltol),
            linear_limb_darkening=linear_limb_darkening,
        )
        return np.asarray(values, dtype=float)

    # This fallback still calls the C++ VBMicrolensing library through its
    # Python binding.  It only exists for an already-installed extension built
    # before the vectorized binding was added; it never falls back to JAX.
    if linear_limb_darkening:
        raise ImportError("Rebuild the native extension for linear limb darkening.")
    if VBMicrolensing is None:
        raise ImportError(
            "FSPL fitting requires the compiled singlelensfitter._vbm_cpp backend "
            "or VBMicrolensing. Rebuild singlelensfitter after installation."
        )
    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = float(tol)
    vbm.RelTol = float(reltol)
    if espl_table_path:
        vbm.LoadESPLTable(str(espl_table_path))
    from .magnification import _espl_binding_values
    values = _espl_binding_values(vbm, u_np, rho_safe)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("VBMicrolensing returned an invalid FSPL magnification.")
    return values


def _cpp_fspl_magnification_gradient(
    u, rho, *, espl_table_path, tol, reltol, linear_limb_darkening=0.0,
    coefficient_gradient=False,
):
    if _vbm_cpp is None:
        raise ImportError("Rebuild the native extension for the FSPL Jacobian.")
    return _vbm_cpp.fspl_magnification_gradient(
        np.asarray(u, dtype=float), max(float(rho), 1e-12), espl_table=espl_table_path,
        tol=tol, reltol=reltol, linear_limb_darkening=linear_limb_darkening,
        coefficient_gradient=coefficient_gradient,
    )


def fspl_magnification_gradient(u, rho, linear_limb_darkening=0.0):
    """Return (A, dA/du, dA/drho, dA/da) for nonnegative separations and I=1-a+a*mu."""
    u = np.asarray(u, dtype=float)
    if np.any(u < 0) or not 1e-12 <= rho <= 100:
        raise ValueError("Require nonnegative u and 1e-12 <= rho <= 100.")
    values = _cpp_fspl_magnification_gradient(
        u.ravel(), rho, espl_table_path=default_espl_table_path(), tol=1e-4, reltol=1e-4,
        linear_limb_darkening=linear_limb_darkening, coefficient_gradient=True,
    )
    return tuple(value.reshape(u.shape) for value in values)


@dataclass
class PSPLFitter:
    """Point-source point-lens fitter using SciPy LM and a native model."""

    maxiter: int = 1000
    tol: float = 1.0e-6
    init_config: Optional[InitConfig] = None

    def __post_init__(self) -> None:
        self.plotter = _make_plotter()
        self._last_fit: Optional[SingleLensFitResult] = None

    def prepare(self, time, flux, ferr):
        """Prepare one light curve for repeated masked fits and objective evaluations."""
        from .prepared import PreparedCurve
        return PreparedCurve(self, time, flux, ferr)

    @property
    def parameter_dimension(self) -> int:
        return 3

    @staticmethod
    def seed_from_fit(fit) -> np.ndarray:
        return np.asarray(fit.params, dtype=float).reshape(-1).copy()

    @staticmethod
    def _decode(z: np.ndarray) -> np.ndarray:
        value = np.asarray(z, dtype=float)
        return np.asarray(
            [value[0], np.exp(np.clip(value[1], -50.0, 50.0)), value[2]],
            dtype=float,
        )

    def _magnification(self, time: np.ndarray, q: np.ndarray) -> np.ndarray:
        return _pspl_magnification(time, q[0], q[1], q[2])

    def _make_fit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        ferr: np.ndarray,
        q: np.ndarray,
        result,
        numpy_result=False,
    ) -> SingleLensFitResult:
        magnification = self._magnification(time, q)
        fs, fb = _solve_fs_fb_numpy(magnification, flux, ferr)
        if not np.isfinite(fs) or not np.isfinite(fb):
            raise RuntimeError("PSPL profiled flux solve is singular.")
        model = fs * magnification + fb
        residual = flux - model
        normalized = residual / ferr
        chi2 = float(np.dot(normalized, normalized))
        asarray = np.asarray if numpy_result else jnp.asarray
        fit = SingleLensFitResult(
            time=time,
            flux=flux,
            ferr=ferr,
            params=asarray(q),
            param_names=("t0", "tE", "u0"),
            chi2=asarray(chi2),
            chi2_dof=asarray(chi2 / max(time.size - 3, 1)),
            fs=asarray(fs),
            fb=asarray(fb),
            model_flux=asarray(model),
            residual=asarray(residual),
            model_kind="pspl",
            optimizer_success=bool(result.success),
            optimizer_status=_optimizer_status(result, maxiter=self.maxiter),
        )
        self._last_fit = fit
        return fit

    def fit(self, time, flux, ferr, p0=None, *, pspl_params=None) -> SingleLensFitResult:
        """Fit PSPL, using the shared automatic multistart when ``p0`` is absent.

        ``pspl_params`` (only its ``t0``) roughly locates the brightening.
        """
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
        if p0 is None:
            seeds = automatic_seeds(
                time_np, flux_np, ferr_np, finite_source=False, pspl_params=pspl_params, config=self.init_config
            )
            best_fit = best_fit_over_seeds(
                lambda seed: self.fit(time_np, flux_np, ferr_np, seed), seeds, "PSPL",
                seed_from_fit=self.seed_from_fit,
            )
            self._last_fit = best_fit
            return best_fit
        return self._fit_from_seed(time_np, flux_np, ferr_np, p0)

    def _fit_from_seed(self, time_np, flux_np, ferr_np, p0, *, numpy_result=False):
        q0 = np.asarray(p0, dtype=float).reshape(-1)
        if q0.size != 3:
            raise ValueError("PSPL p0 must contain (t0, tE, u0).")
        if not np.all(np.isfinite(q0)) or q0[1] == 0.0:
            raise ValueError("PSPL p0 must be finite and have nonzero tE.")
        # LM's step-size convergence test depends on the coordinate norm.
        # Center t0 so absolute Julian dates cannot trigger premature success.
        time_origin = q0[0]
        fit_time = time_np - time_origin
        z0 = np.asarray([0.0, np.log(abs(q0[1])), q0[2]], dtype=float)

        profile = _FluxProfile(flux_np, ferr_np)
        cached_z = None
        cached_model = None

        def model(z):
            nonlocal cached_z, cached_model
            if cached_z is None or not np.array_equal(z, cached_z):
                t0, te, u0 = self._decode(z)
                te = max(te, 1e-12)
                tau = (fit_time-t0)/te
                u2 = np.maximum(tau*tau + u0*u0, 1e-24)
                root = np.sqrt(u2)*np.sqrt(u2+4)
                magnification = (u2+2)/root
                cached_model = magnification, tau, u2, root, te
                cached_z = z.copy()
            return cached_model

        def residual(z):
            magnification = model(z)[0]
            fs, fb = profile.solve(magnification)
            values = (flux_np - (fs * magnification + fb)) / ferr_np
            return values if np.all(np.isfinite(values)) else np.full_like(values, 1e50)

        def jacobian(z):
            magnification, tau, u2, root, te = model(z)
            # dA/du divided by u for the rectilinear trajectory.
            slope = -8/(root*u2*(u2+4))
            gradient = np.empty((time_np.size,3), order="F")
            gradient[:,0] = -slope*tau/te
            gradient[:,1] = -slope*tau*tau
            gradient[:,2] = slope*z[2]
            value = profile.jacobian(magnification, gradient)
            return value if np.isfinite(value).all() else np.zeros_like(value)

        result = _run_scipy_lm(
            residual,
            z0,
            maxiter=self.maxiter,
            tol=self.tol,
            jac=jacobian,
        )
        q = self._decode(np.asarray(result.x, dtype=float))
        q[0] += time_origin
        return self._make_fit(time_np, flux_np, ferr_np, q, result, numpy_result=numpy_result)

    def evaluate_fixed(self, time, flux, ferr, p0) -> SingleLensFitResult:
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
        q = np.asarray(p0, dtype=float).reshape(-1)
        if q.size != 3 or not np.all(np.isfinite(q)) or q[1] == 0.0:
            raise ValueError("PSPL fixed parameters must contain finite (t0, tE, u0).")
        return self._make_fit(
            time_np,
            flux_np,
            ferr_np,
            q,
            SimpleNamespace(
                fixed=True,
                success=True,
                status=0,
                nfev=0,
                message="fixed_parameters",
            ),
        )

    def fit_fixed_model(self, time, flux, ferr, p0, *, model_kind=None):
        """Evaluate fixed nonlinear parameters through the common fitter API."""
        if model_kind not in {None, "pspl"}:
            raise ValueError(f"PSPLFitter cannot evaluate model_kind={model_kind!r}.")
        return self.evaluate_fixed(time, flux, ferr, p0)

    def plot_lc(self, **kwargs):
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)


@dataclass
class FSPLFitter:
    """Finite-source point-lens fitter using C++ magnification and SciPy LM."""

    maxiter: int = 1000
    tol: float = 1.0e-6
    magnification_tol: float = 1.0e-4
    magnification_reltol: float = 1.0e-4
    espl_table_path: Optional[str] = None

    linear_limb_darkening: float = 0.0
    fit_limb_darkening: bool = False
    init_config: Optional[InitConfig] = None

    def __post_init__(self) -> None:
        if not 0 <= self.linear_limb_darkening <= 1:
            raise ValueError("linear_limb_darkening must be in [0, 1]")
        self.plotter = _make_plotter()
        self._last_fit: Optional[SingleLensFitResult] = None
        if self.espl_table_path is None:
            self.espl_table_path = default_espl_table_path()

    def prepare(self, time, flux, ferr):
        """Prepare one light curve for repeated masked fits and objective evaluations."""
        from .prepared import PreparedCurve
        return PreparedCurve(self, time, flux, ferr)

    @property
    def parameter_dimension(self) -> int:
        return 5 if self.fit_limb_darkening else 4

    @staticmethod
    def seed_from_fit(fit) -> np.ndarray:
        params = np.asarray(fit.params, dtype=float).reshape(-1)
        if params.size not in (4, 5):
            raise ValueError("Expected four FSPL parameters and an optional brightness coefficient.")
        seed = params.copy()
        seed[3] = np.log(max(abs(seed[3]), 1e-12))
        return seed

    def _seed(self, p0):
        seed = np.asarray(p0, dtype=float).reshape(-1)
        if self.fit_limb_darkening and seed.size == 4:
            seed = np.append(seed, self.linear_limb_darkening)
        if seed.size != self.parameter_dimension or not np.all(np.isfinite(seed)) or seed[1] == 0:
            raise ValueError("Expected finite FSPL seed (t0, tE, u0, logrho[, a]) with nonzero tE.")
        if seed.size == 5 and not 0 <= seed[4] <= 1:
            raise ValueError("The initial limb-darkening coefficient must be in [0, 1].")
        return seed

    @staticmethod
    def _decode(z: np.ndarray) -> np.ndarray:
        value = np.asarray(z, dtype=float).copy()
        value[1] = np.exp(np.clip(value[1], -50., 50.))
        if value.size == 5:
            gamma = value[4]
            value[4] = 3*gamma/(2+gamma)
        return value

    def _magnification(self, time: np.ndarray, q: np.ndarray) -> np.ndarray:
        t0, tE, u0, logrho = q[:4]
        if hasattr(_vbm_cpp, "fspl_light_curve"):
            return _vbm_cpp.fspl_light_curve(
                time, max(_espl_radius(logrho), 1e-12), t0, tE, u0, self.espl_table_path,
                self.magnification_tol, self.magnification_reltol,
                q[4] if q.size == 5 else self.linear_limb_darkening,
            )
        u = np.sqrt(((time - t0) / max(abs(tE), 1.0e-12)) ** 2 + u0 * u0)
        rho = _espl_radius(logrho)
        return _cpp_fspl_magnification(
            u,
            rho,
            linear_limb_darkening=q[4] if q.size == 5 else self.linear_limb_darkening,
            espl_table_path=self.espl_table_path,
            tol=self.magnification_tol,
            reltol=self.magnification_reltol,
        )

    def _make_fit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        ferr: np.ndarray,
        q: np.ndarray,
        result,
        numpy_result=False,
    ) -> SingleLensFitResult:
        magnification = self._magnification(time, q)
        fs, fb = _solve_fs_fb_numpy(magnification, flux, ferr)
        if not np.isfinite(fs) or not np.isfinite(fb):
            raise RuntimeError("FSPL profiled flux solve is singular.")
        model = fs * magnification + fb
        residual = flux - model
        normalized = residual / ferr
        chi2 = float(np.dot(normalized, normalized))
        params = q.copy()
        params[3] = _espl_radius(q[3])
        a = q[4] if q.size == 5 else self.linear_limb_darkening
        evaluation_seed = np.r_[q[:4], a]
        asarray = np.asarray if numpy_result else jnp.asarray
        fit = SingleLensFitResult(
            time=time,
            flux=flux,
            ferr=ferr,
            params=asarray(params),
            param_names=("t0", "tE", "u0", "rho") + (("a",) if q.size == 5 else ()),
            linear_limb_darkening=float(a),
            chi2=asarray(chi2),
            chi2_dof=asarray(chi2 / max(time.size - q.size, 1)),
            fs=asarray(fs),
            fb=asarray(fb),
            model_flux=asarray(model),
            residual=asarray(residual),
            raw_params=asarray(q),
            model_kind="fspl",
            model_evaluator=lambda values, q=evaluation_seed: self._magnification(
                np.asarray(values, dtype=float), q
            ),
            optimizer_success=bool(result.success),
            optimizer_status=_optimizer_status(
                result, maxiter=self.maxiter, detail="compiled_magnification",
                method="trf" if self.fit_limb_darkening else "lm"
            ),
        )
        self._last_fit = fit
        return fit

    def fit_fixed_model(self, time, flux, ferr, p0, *, model_kind=None):
        """Evaluate fixed nonlinear parameters through the common fitter API."""
        if model_kind not in {None, "fspl"}:
            raise ValueError(f"FSPLFitter cannot evaluate model_kind={model_kind!r}.")
        return self.evaluate_fixed(time, flux, ferr, p0)

    def _fit_from_seed(
        self,
        time_np: np.ndarray,
        flux_np: np.ndarray,
        ferr_np: np.ndarray,
        p0,
        *, numpy_result=False,
    ) -> SingleLensFitResult:
        q0 = self._seed(p0)

        # Keep the optimizer's t0 coordinate close to zero.  SciPy's default
        # finite-difference step is proportional to max(1, abs(x0)); using an
        # absolute BJD/HJD here can therefore make the first t0 derivative
        # many days wide.  The old explicit relative step had the opposite
        # problem for coordinates at zero (notably logrho=0), where it fell
        # back to a machine-scale perturbation.  Evaluate the model in the
        # translated time frame and restore the public t0 below.
        time_origin = float(q0[0])
        fit_time = time_np - time_origin
        z0 = q0.copy()
        z0[:2] = [0., np.log(abs(q0[1]))]
        if self.fit_limb_darkening:
            # Optimize the linear mixing fraction of the unit-flux bases.
            z0[4] = 2*q0[4]/(3-q0[4])

        profile = _FluxProfile(flux_np, ferr_np)
        has_native_partials = hasattr(_vbm_cpp, "fspl_magnification_gradient")
        cached_z = None
        cached_partials = None

        def model_partials(z, q):
            # SciPy requests the residual and Jacobian at the same accepted step.
            nonlocal cached_z, cached_partials
            if cached_z is None or not np.array_equal(z, cached_z):
                tau = (fit_time - q[0]) / max(q[1], 1e-12)
                u = np.sqrt(tau*tau + q[2]*q[2])
                rho = _espl_radius(q[3])
                partials = _cpp_fspl_magnification_gradient(
                    u, rho, espl_table_path=self.espl_table_path,
                    tol=self.magnification_tol, reltol=self.magnification_reltol,
                    linear_limb_darkening=q[4] if q.size == 5 else self.linear_limb_darkening,
                    coefficient_gradient=self.fit_limb_darkening,
                )
                cached_partials = tau, u, rho, partials
                cached_z = z.copy()
            return cached_partials

        def residual(z):
            q = self._decode(z)
            try:
                magnification = (model_partials(z, q)[3][0] if has_native_partials
                                 else self._magnification(fit_time, q))
                fs, fb = profile.solve(magnification)
                if not np.isfinite(fs) or not np.isfinite(fb):
                    raise ValueError
                values = (flux_np - (fs * magnification + fb)) / ferr_np
                if not np.all(np.isfinite(values)):
                    raise ValueError
                return values
            except Exception:
                return np.full(time_np.size, 1.0e50, dtype=float)

        def jacobian(z):
            q = self._decode(z)
            try:
                tau, u, rho, partials = model_partials(z, q)
                magnification, dA_du, dA_drho = partials[:3]
                slope = dA_du / np.maximum(u, 1e-12)
                gradient = np.empty((u.size, q.size), dtype=float, order="F")
                gradient[:, 0] = -slope * tau / max(q[1], 1e-12)
                gradient[:, 1] = -slope * tau * tau
                gradient[:, 2] = slope * q[2]
                gradient[:, 3] = dA_drho * rho if rho > 1e-12 and -50 < q[3] < 10 else 0.
                if self.fit_limb_darkening:
                    gradient[:, 4] = partials[3] * (6/(2+z[4])**2)
                value = profile.jacobian(magnification, gradient)
                if not np.isfinite(value).all():
                    raise ValueError
                return value
            except Exception:
                return np.zeros((time_np.size, self.parameter_dimension), dtype=float)

        def run_once(start: np.ndarray):
            if self.fit_limb_darkening:
                result = least_squares(
                    residual, start, jac=jacobian, method="trf", x_scale="jac",
                    bounds=([-np.inf] * 4 + [0.], [np.inf] * 4 + [1.]),
                    xtol=self.tol, ftol=self.tol, gtol=self.tol, max_nfev=self.maxiter,
                )
            else:
                result = _run_scipy_lm(
                    residual,
                    np.asarray(start, dtype=float),
                    maxiter=self.maxiter,
                    tol=self.tol,
                    jac=jacobian,
                )
            q = self._decode(np.asarray(result.x, dtype=float))
            q[0] += time_origin
            values = result.fun if hasattr(result, "fun") else residual(result.x)
            return result, q, float(np.dot(values, values))

        result, q, objective = run_once(z0)
        retry_count = 0
        initial_rho = np.exp(np.clip(q0[3], -50.0, 10.0))
        fitted_rho = _espl_radius(q[3])
        # A finite-source fit can fall into the nearly-point-source basin when
        # the starting radius is on the large side.  Keep one canonical LM
        # implementation, but retry from a smaller radius when that diagnostic
        # is present; this is a basin rescue, not a second fitter/backend.
        if fitted_rho < 0.5 * initial_rho and np.isfinite(objective):
            retry = np.asarray(z0, dtype=float).copy()
            retry[3] -= np.log(10.0 / 3.0)
            retry_result, retry_q, retry_objective = run_once(retry)
            if np.isfinite(retry_objective) and retry_objective < objective:
                result, q, objective = retry_result, retry_q, retry_objective
                retry_count = 1
        if retry_count:
            result.message = f"{result.message}; basin_retry={retry_count}"
        return self._make_fit(time_np, flux_np, ferr_np, q, result, numpy_result=numpy_result)

    def _automatic_initial_guesses(
        self,
        time_np: np.ndarray,
        flux_np: np.ndarray,
        ferr_np: np.ndarray,
        *,
        pspl_params=None,
        observed_scale=None,
    ) -> tuple[np.ndarray, ...]:
        """Shared multistart set (duration-grid anchors + degeneracy family)."""
        return automatic_seeds(
            time_np,
            flux_np,
            ferr_np,
            finite_source=True,
            config=self.init_config,
            pspl_params=pspl_params,
            observed_scale=observed_scale,
            magnification_tol=self.magnification_tol,
            magnification_reltol=self.magnification_reltol,
            espl_table_path=self.espl_table_path,
        )

    def fit(
        self,
        time,
        flux,
        ferr,
        p0=None,
        *,
        pspl_params=None,
        observed_scale=None,
    ) -> SingleLensFitResult:
        """Fit FSPL, using the observed-duration seed grid when ``p0`` is absent."""
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
        if p0 is None:
            seeds = self._automatic_initial_guesses(
                time_np,
                flux_np,
                ferr_np,
                pspl_params=pspl_params,
                observed_scale=observed_scale,
            )
            def fit_one(seed):
                return self._fit_from_seed(time_np, flux_np, ferr_np, seed)

            best_fit = best_fit_over_seeds(fit_one, seeds, "FSPL", seed_from_fit=self.seed_from_fit)

            def try_alternate(current, seed):
                return best_fit_over_seeds(fit_one, [seed], "FSPL", incumbent=current,
                                           seed_from_fit=self.seed_from_fit)

            # A transit straddling an observing gap can fit the visible limb
            # with more than one peak time. Probe farther into the missing half.
            t0, tE, u0, rho = np.asarray(best_fit.params)[:4]
            dt = time_np - t0
            crossing = tE * rho
            left = np.any((dt >= -crossing) & (dt <= 0))
            right = np.any((dt >= 0) & (dt <= crossing))
            if abs(u0) < rho and left != right:
                seed = self.seed_from_fit(best_fit)
                seed[0] += (0.5 if left else -0.5) * crossing
                best_fit = try_alternate(best_fit, seed)

            # Re-evaluate coverage at the best peak. If only one wing is
            # sampled within tE, try central and grazing chords ending near
            # the nearest epoch. Offset by half a cadence to avoid starting
            # a derivative-based fit exactly at singular limb contact.
            t0, tE = np.asarray(best_fit.params)[:2]
            dt = time_np - t0
            left_wing = np.any((dt >= -tE) & (dt <= 0))
            right_wing = np.any((dt >= 0) & (dt <= tE))
            if left_wing != right_wing:
                spacings = np.diff(np.unique(time_np))
                half_cadence = 0.5 * np.median(spacings) if spacings.size else 0.0
                duration = np.min(np.abs(dt)) + half_cadence
                for impact_ratio in (0.05, 0.8):
                    radius = duration / (tE * np.sqrt(1 - impact_ratio**2))
                    seed = [t0, tE, impact_ratio * radius, np.log(radius)]
                    if self.fit_limb_darkening:
                        seed.append(best_fit.linear_limb_darkening)
                    best_fit = try_alternate(best_fit, seed)
            self._last_fit = best_fit
            return best_fit
        return self._fit_from_seed(time_np, flux_np, ferr_np, p0)

    def evaluate_fixed(self, time, flux, ferr, p0) -> SingleLensFitResult:
        time_np, flux_np, ferr_np = _fit_arrays(time, flux, ferr)
        q = self._seed(p0)
        fit = self._make_fit(
            time_np,
            flux_np,
            ferr_np,
            q,
            SimpleNamespace(
                fixed=True,
                success=True,
                status=0,
                nfev=0,
                message="fixed_parameters",
            ),
        )
        return fit

    def plot_lc(self, **kwargs):
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_lc(self._last_fit, **kwargs)

    def plot_residual(self, **kwargs):
        if self._last_fit is None:
            raise RuntimeError("No fit has been run yet.")
        return self.plotter.plot_residual(self._last_fit, **kwargs)


__all__ = [
    "PSPLFitter",
    "FSPLFitter",
    "PSPLParallaxFitter",
    "FSPLParallaxFitter",
]
