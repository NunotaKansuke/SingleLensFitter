"""Full-cadence RMDC integrations. Set RMDC_ML_ROOT and RMDC_EPHEMERIS_PATH."""

import os
from pathlib import Path

import numpy as np
import pytest

from singlelensfitter import (
    Ephemeris, FSPLFitter, FSPLParallaxFitter, PSPLFitter, PSPLParallaxFitter, TimeSpec,
)
from singlelensfitter.parallax_backend import default_earth_ephemeris, load_vbm_satellite_ephemeris


@pytest.fixture(scope="module")
def rmdc():
    if "RMDC_ML_ROOT" not in os.environ:
        pytest.skip("Set RMDC_ML_ROOT to the RMDC26 machine-learning parquet directory")
    import pyarrow.parquet as pq

    root = Path(os.environ["RMDC_ML_ROOT"])
    ids = [3008, 3745, 5177, 5246, 6152]
    epochs = pq.read_table(root / "RMDC26_ML_Data_epoch.parquet", columns=["epoch_id", "bjd"])
    bjd = dict(zip(epochs["epoch_id"].to_pylist(), epochs["bjd"].to_pylist()))
    rows = pq.read_table(root / "RMDC26_ML_Data_obs.parquet",
                         columns=["event_id", "epoch_id", "flux_uJy", "flux_err_uJy"],
                         filters=[("event_id", "in", ids), ("filt", "=", "F146"),
                                  ("saturation_flag", "=", 0)], use_threads=False)
    meta = pq.read_table(root / "RMDC26_ML_Data_meta.parquet",
                         columns=["event_id", "ra_deg", "dec_deg"],
                         filters=[("event_id", "in", ids)])
    t = np.array([bjd[i] for i in rows["epoch_id"].to_pylist()]) - 2460000
    f, e = (rows[key].to_numpy() for key in ("flux_uJy", "flux_err_uJy"))
    curves = {}
    for row in meta.to_pylist():
        event = row["event_id"]
        use = ((rows["event_id"].to_numpy() == event) & np.isfinite(t) &
               np.isfinite(f) & np.isfinite(e) & (e > 0))
        curves[event] = (t[use], f[use], e[use]), (row["ra_deg"], row["dec_deg"])
    return curves


def check_fit(fitter, fit, curve):
    time, flux, ferr = curve
    assert fit.optimizer_success, fit.optimizer_status
    assert np.all(np.isfinite(fit.params))
    np.testing.assert_allclose(fit.residual, flux - fit.model_flux, atol=1e-10)
    np.testing.assert_allclose(fit.chi2, np.sum((fit.residual / ferr)**2), rtol=1e-10)
    replay = fitter.evaluate_fixed(time, flux, ferr, fitter.seed_from_fit(fit))
    np.testing.assert_allclose(replay.model_flux, fit.model_flux, rtol=1e-9)
    print(f"n={len(time)} {fit.model_kind} chi2={float(fit.chi2):.6f} params={fit.params}")


@pytest.mark.parametrize("event", [3008, 3745, 5177, 5246, 6152])
def test_rmdc_automatic_point_and_finite_source(rmdc, event):
    curve, _ = rmdc[event]
    t, flux, ferr = curve
    point, finite = PSPLFitter(), FSPLFitter()
    p = point.fit(*curve)
    check_fit(point, p, curve)
    # Independent weighted least-squares solution at the returned geometry.
    t0, te, u0 = p.params.tolist()
    u = np.hypot((t - t0) / te, u0)
    a = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    design = np.column_stack([a, np.ones_like(a)])
    coeff = np.linalg.lstsq(design / ferr[:, None], flux / ferr, rcond=None)[0]
    np.testing.assert_allclose(design @ coeff, p.model_flux, rtol=1e-8)
    f = finite.fit(*curve)
    check_fit(finite, f, curve)
    if event in (3008, 3745):
        assert float(f.chi2) < float(p.chi2)
    if event == 5246:
        # Re-score the archived table-based solution with the current kernel.
        archived = [2986.34809, 97.2579208, -0.417508412, np.log(0.434797270)]
        assert float(f.chi2) <= float(finite.evaluate_fixed(*curve, archived).chi2)
    if event == 3745:
        # A separate fit started at injection geometry reaches chi2=46181.767.
        # The peak falls immediately before the start of a season.
        assert float(f.chi2) <= 46182
    if event == 3008:
        assert float(p.chi2 - f.chi2) > 100
        # Known in-season transit, within one F146 exposure spacing.
        assert abs(float(f.params[0]) - 3106.479372621) < 0.012


@pytest.fixture(scope="module")
def space_options():
    if "RMDC_EPHEMERIS_PATH" not in os.environ:
        pytest.skip("Set RMDC_EPHEMERIS_PATH to the RMDC geocentric satellite table")
    spec = TimeSpec("jd", offset=2460000)
    earth = default_earth_ephemeris(time_spec=spec)
    sat = load_vbm_satellite_ephemeris(os.environ["RMDC_EPHEMERIS_PATH"], time_spec=spec)
    er = np.column_stack([np.interp(sat.time, earth.time, x) for x in earth.position_au.T])
    obs = Ephemeris(sat.time, er + sat.position_au, origin="explicit_reference", time_spec=spec)
    return dict(geometry="space", observer_convention="gulls", earth_ephemeris=earth,
                   satellite_or_observer_ephemeris=obs, reference_ephemeris=earth,
                   time_spec=spec, max_piE=10)


@pytest.mark.parametrize("event, reference_chi2", [(5177, 46468.8545), (5246, 46954.40136)])
def test_rmdc_space_parallax(rmdc, space_options, event, reference_chi2):
    curve, coords = rmdc[event]
    t, flux, _ = curve
    tref = float(t[np.argmax(flux)])
    point = PSPLParallaxFitter(*coords, tref, **space_options)
    fit = point.fit(*curve)
    check_fit(point, fit, curve)
    # Archived run of this event at the same max-flux reference epoch.
    assert abs(float(fit.chi2) - reference_chi2) < 0.1
    finite = FSPLParallaxFitter(*coords, tref, **space_options)
    seed = np.insert(point.seed_from_fit(fit), 3, np.log(0.001))
    fs = finite.fit(*curve, seed)
    check_fit(finite, fs, curve)
    assert float(fs.chi2) <= float(fit.chi2) + 0.1


def test_rmdc_automatic_fspl_parallax_transit(rmdc, space_options):
    curve, coords = rmdc[3008]
    t, flux, _ = curve
    finite = FSPLParallaxFitter(*coords, float(t[np.argmax(flux)]), **space_options)
    fit = finite.fit(*curve)
    check_fit(finite, fit, curve)
    # The zero-parallax solution on these data has chi2 = 47150.913.
    assert float(fit.chi2) <= 47151


@pytest.mark.parametrize("event", [3008, 3745])
def test_rmdc_profiled_limb_darkening(rmdc, event):
    curve, _ = rmdc[event]
    uniform = FSPLFitter()
    baseline = uniform.fit(*curve)
    seed = uniform.seed_from_fit(baseline)
    darkened = FSPLFitter(linear_limb_darkening=.3, tol=1e-9)
    fixed = darkened.fit(*curve, seed)
    check_fit(darkened, fixed, curve)
    joint = FSPLFitter(linear_limb_darkening=.3, fit_limb_darkening=True, tol=1e-9)
    fitted = joint.fit(*curve, seed)
    check_fit(joint, fitted, curve)
    assert 0 <= fitted.linear_limb_darkening <= 1
    assert fitted.param_names[-1] == "a"
    assert float(fitted.chi2) <= min(float(baseline.chi2), float(fixed.chi2)) + .01
