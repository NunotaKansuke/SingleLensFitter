"""Fit matched benchmark photometry using singlens and JAX autodiff."""
import csv
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
import singlens

from methods import photometry, score_reference


def run(rho, a, workflows=False):
    time, flux, error, seed = photometry(rho, a)
    model = singlens.Model(source=("linear" if a else "uniform") if rho else "point")
    times, values, errors = map(jnp.asarray, (time - seed[0], flux, error))
    weights = errors**-2
    weights /= weights.sum()
    centered_flux = values - weights @ values

    def residual(z):
        params = singlens.Parameters(t0=z[0], t_e=jnp.exp(z[1]), u0=z[2],
                                     rho=jnp.exp(z[3]) if rho else None,
                                     a1=a if a else None)
        mag = model.magnification(times, params)
        centered = mag - weights @ mag
        weighted = weights * centered
        source = (weighted @ centered_flux) / (weighted @ centered)
        return (centered_flux - source * centered) / errors

    fun = jax.jit(residual)
    jac = jax.jit(jax.jacfwd(residual))
    start = np.r_[0., np.log(seed[1]), seed[2], [np.log(seed[3])] if rho else []]
    tic = perf_counter()
    fun(start).block_until_ready()
    jac(start).block_until_ready()
    warmup_ms = 1000 * (perf_counter() - tic)
    options = dict(method="lm", x_scale=1., ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=1000)
    if workflows:
        lower, upper = np.full(start.size, -np.inf), np.full(start.size, np.inf)
        if rho:
            lower[3], upper[3] = np.log([1e-6, 10.])
        options = dict(method="trf", bounds=(lower, upper), x_scale="jac",
                       ftol=1e-10, xtol=1e-10, gtol=1e-8, max_nfev=400)
    for repeat in range(10):
        tic = perf_counter()
        result = least_squares(fun, start, jac=jac, **options)
        ms = 1000 * (perf_counter() - tic)
        if not repeat:
            continue
        q = result.x.copy()
        q[0] += seed[0]
        q[1] = np.exp(q[1])
        if rho:
            q[3] = np.exp(q[3])
        yield [rho, a, "singlens TRF" if workflows else "singlens autodiff", repeat, ms, result.fun @ result.fun,
               score_reference(time, flux, error, q, a), *q[:3], q[3] if rho else "",
               "ok" if result.success else result.message, result.status, result.nfev, warmup_ms]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["rho_true", "a", "method", "repeat", "ms", "chi2", "reference_chi2",
                     "t0", "tE", "u0", "rho_fit", "status", "lm_status", "nfev", "warmup_ms"])
    for rho, a in [(0, 0), (.024, 0), (.024, .5), (.24, 0), (.24, .5)]:
        writer.writerows(run(rho, a, args.workflows))
        sys.stdout.flush()
