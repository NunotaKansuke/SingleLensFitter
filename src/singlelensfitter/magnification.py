from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import jit

_mag_fspl = {}


def _get_fspl_disk(N_fft: int = 1024):
    """Return a cached microjax uniform-disk magnifier.

    ``N_fft`` is exposed for fast, peak-only exploratory fits.  The default
    keeps the established 1024-point accuracy setting.
    """
    global _mag_fspl
    if _mag_fspl is None:
        _mag_fspl = {}
    if N_fft not in _mag_fspl:
        try:
            from microjax.fastlens import fspl_disk
        except ImportError as exc:
            raise ImportError(
                "microjax is required for FSPL magnification. Install it from "
                "https://github.com/ShotaMiyazaki94/microjax before using FSPL fitters."
            ) from exc
        _mag_fspl[N_fft] = fspl_disk(N_fft=N_fft)
    return _mag_fspl[N_fft]

@jit
def A_pspl_from_u(u: jnp.ndarray) -> jnp.ndarray:
    # u can be scalar or array
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))

def A_fspl_from_u(u, rho, *, N_fft: int = 1024):
    return _get_fspl_disk(N_fft=N_fft).A(u, rho)


from functools import lru_cache as _lru_cache
from pathlib import Path

@_lru_cache(maxsize=1)
def _native_fspl_magnifier():
    """Return a process-local VBMicrolensing ESPL evaluator."""
    try:
        import VBMicrolensing
    except ImportError as exc:  # pragma: no cover - required package dependency
        raise ImportError(
            "VBMicrolensing>=5.5 is required for the native FSPL detector."
        ) from exc
    magnifier = VBMicrolensing.VBMicrolensing()
    magnifier.Tol = 1.0e-4
    magnifier.RelTol = 1.0e-4
    table = Path(VBMicrolensing.__file__).resolve().parent / "data" / "ESPL.tbl"
    if table.is_file():
        magnifier.LoadESPLTable(str(table))
    return magnifier


def _espl_binding_values(magnifier, u, rho):
    """Scalar VBM binding with limb integration for installations without the vector API."""
    from scipy.integrate import quad

    def value(separation):
        z = abs(float(separation)) / rho
        distance = abs(z - 1)
        if z >= 1.15:
            return magnifier.ESPLMag2(abs(float(separation)), rho)

        def boundary(angle):
            half = 2 * np.sin(0.5 * angle)**2
            squared = (z - 1)**2 + 2 * z * half
            return ((1 - z) + z * half) * np.sqrt(1 + 4 / (rho**2 * squared))

        points = [distance] if distance else None
        exact = quad(boundary, 0, np.pi, points=points, epsrel=1e-10)[0] / np.pi
        if z <= 1.05:
            return exact
        t = (z - 1.05) / 0.10
        return exact + t*t*(3 - 2*t)*(magnifier.ESPLMag2(abs(float(separation)), rho) - exact)

    return np.asarray([value(x) for x in u], dtype=float)


def _native_fspl_magnification(u: np.ndarray, rho: float) -> np.ndarray:
    """Evaluate an exact uniform-source curve through the native C++ backend."""
    magnifier = _native_fspl_magnifier()
    rho_safe = max(float(rho), 1.0e-12)
    if not rho_safe <= 100.0:
        raise ValueError("ESPL requires rho <= 100")
    values = _espl_binding_values(magnifier, u, rho_safe)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise RuntimeError("VBMicrolensing returned an invalid FSPL magnification.")
    return values
