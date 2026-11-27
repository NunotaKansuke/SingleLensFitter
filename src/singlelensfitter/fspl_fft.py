"""Blind finite-source FFT search in crossing time and scaled impact parameter."""
from functools import lru_cache

import numpy as np

from .auto_init import filter_pspl_fft_candidates
from .pspl_fft import PSPLFFTScanner, pspl_excess_magnification


@lru_cache(maxsize=32)
def _shape(rho, impact, coefficient, table, tol, reltol):
    from .fitters import _cpp_fspl_magnification, default_espl_table_path

    # Include limb contact exactly when interpolating the template in |t-t0|/t_star.
    x = np.unique(np.r_[0., np.geomspace(1e-5, 1e6, 8192),
                        np.sqrt(max(0., 1-impact**2))])
    u = rho*np.hypot(x, impact)
    excess = _cpp_fspl_magnification(u, rho, espl_table_path=table or default_espl_table_path(), tol=tol,
                                     reltol=reltol, linear_limb_darkening=coefficient) - 1.
    return x, excess


def fft_seeds(time, flux, ferr, fitter, config):
    if len(time) < 2 or np.ptp(time) == 0:
        return ()
    scanner = PSPLFFTScanner(grid_dt=config.auto_init_fft_grid_dt,
                             max_grid_points=config.auto_init_fft_max_grid_points,
                             fft_workers=1)
    limit = int(config.auto_init_fspl_template_top_k)
    if limit < 1:
        raise ValueError("auto_init_fspl_template_top_k must be positive.")
    span = np.ptp(time)
    dt = scanner.grid_dt or np.median(np.diff(np.unique(time)))
    dt = max(dt, span / (scanner.max_grid_points - 1) * (1 + 1e-12))
    shapes = ((0.03, 1.5), (0.1, 0.25), (1., 0.8), (10., 0.3))
    radii = np.array([rho for rho, _ in shapes])
    impacts = np.array([rho * impact for rho, impact in shapes])
    curves = [_shape(rho, impact, fitter.linear_limb_darkening,
                     fitter.espl_table_path, fitter.magnification_tol,
                     fitter.magnification_reltol) for rho, impact in shapes]
    point_u0 = 0.5
    point_ranked = []
    ranked = []
    seeds = {}
    data = None
    for width in (0.05, 0.25, 1.25, 6.25, 31.25):
        # Broad templates need fewer shifts. LM subsequently uses every epoch.
        spacing = max(dt, min(width / 32, span / 64))
        if data is None or data.grid_dt != spacing:
            data = scanner._prepare(time, flux, ferr, grid_dt=spacing)
        lags = data.lags / width
        templates = [np.interp(lags, x, excess, right=0.) for x, excess in curves]
        templates.append(pspl_excess_magnification(data.lags, point_u0, width))
        profiles = scanner._profile_template_batch(
            data, np.stack(templates), np.r_[impacts, point_u0],
            np.r_[width / radii, width / point_u0],
        )
        # The point-source profile has no radius. Start its continuous FSPL
        # fit outside transit with a positive, freely varying source radius.
        for row, (profile, rho) in enumerate(zip(profiles, np.r_[radii, point_u0 / 4])):
            for index in scanner._peak_indices(profile, 1):
                candidate = profile.candidate(index)
                bank = ranked if row < len(radii) else point_ranked
                bank.append(candidate)
                seeds[candidate] = np.array([candidate.t0, candidate.tE,
                                             candidate.u0, np.log(rho)])
    ranked.sort(key=lambda c: c.delta_chi2, reverse=True)
    selected = filter_pspl_fft_candidates(ranked, time_np=time, flux_np=flux,
                                          limit=limit, config=config)
    point_ranked.sort(key=lambda c: c.delta_chi2, reverse=True)
    point = filter_pspl_fft_candidates(point_ranked, time_np=time, flux_np=flux,
                                       limit=1, config=config)
    if point and (not selected or point[0].delta_chi2 >= selected[0].delta_chi2):
        selected.extend(point)
    return tuple(seeds[c] for c in selected)
