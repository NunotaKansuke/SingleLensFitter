"""Differentiable native FSPL magnification for JAX trajectories."""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=1)
def _register_partials():
    from . import _vbm_cpp
    name = "singlelensfitter_fspl_partials_f64"
    jax.ffi.register_ffi_target(name, _vbm_cpp.fspl_ffi_capsule(), platform="cpu")
    return name


@lru_cache(maxsize=8)
def _magnifier(table_path, backend):
    from . import _vbm_cpp

    if table_path and hasattr(_vbm_cpp, "fspl_ffi_capsule") and backend == "cpu":
        target = _register_partials()

        def partials(u, rho, a):
            spec = jax.ShapeDtypeStruct((4*u.size,), u.dtype)
            call = jax.ffi.ffi_call(target, spec, vmap_method="sequential")
            return call(u.ravel(), rho, a, table=table_path).reshape(4, *u.shape)
    else:
        def evaluate(u, rho, a):
            shape = u.shape
            if not (0 < rho <= 100 and 0 <= a <= 1 and np.isfinite(u).all()):
                return np.full((4, *shape), np.nan)
            values = _vbm_cpp.fspl_magnification_gradient(
                np.asarray(u).ravel(), float(rho), table_path, 1e-4, 1e-4, float(a), True)
            return np.stack(values).reshape(4, *shape)

        def partials(u, rho, a):
            spec = jax.ShapeDtypeStruct((4, *u.shape), u.dtype)
            return jax.pure_callback(evaluate, spec, u, rho, a, vmap_method="sequential")

    @jax.custom_jvp
    def magnification(u, rho, a):
        return partials(u, rho, a)[0]

    @magnification.defjvp
    def differential(primals, tangents):
        value, du, dr, da = partials(*primals)
        return value, du*tangents[0] + dr*tangents[1] + da*tangents[2]

    return magnification


def jax_fspl_magnification(u, rho, linear_limb_darkening=0., *, espl_table_path=None):
    """Evaluate A(|u|, rho, a) with first derivatives, for a scalar radius and coefficient.

    A real separation or complex position array determines the output shape.
    The CPU FFI is used when compiled into the extension. Fluxes, trajectory,
    priors and sampling coordinates belong to the calling model.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Native FSPL requires jax_enable_x64=True")
    if espl_table_path is None:
        from .parallax_backend import default_espl_table_path
        espl_table_path = default_espl_table_path()
    path = str(espl_table_path) if espl_table_path is not None else None
    rho = jnp.asarray(rho, dtype=jnp.float64)
    coefficient = jnp.asarray(linear_limb_darkening, dtype=jnp.float64)
    if rho.ndim or coefficient.ndim:
        raise ValueError("rho and linear_limb_darkening must be scalars")
    u = jnp.asarray(jnp.abs(jnp.asarray(u)), dtype=jnp.float64)
    return _magnifier(path, jax.default_backend())(u, rho, coefficient)
