import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("singlelensfitter._parallax_cpp")

from singlelensfitter import (
    Ephemeris,
    PSPLParallaxFitter,
    ParallaxEvaluator,
    TimeSpec,
)
from singlelensfitter.parallax_backend import _profile_fluxes, parallax_effect_score


def _linear_ephemeris(origin="sun", time_spec=TimeSpec()):
    time = np.asarray([0.0, 1.0, 2.0])
    position = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.1, 0.2], [0.0, 0.4, 0.8]])
    return Ephemeris(time, position, origin=origin, time_spec=time_spec)


def test_native_basis_and_canonical_sign_at_reference_epoch():
    earth = _linear_ephemeris(time_spec=TimeSpec("hjd"))
    evaluator = ParallaxEvaluator(
        np.asarray([0.0, 1.0, 2.0]), np.ones(3), np.ones(3),
        ra_deg=0.0, dec_deg=0.0, tref=1.0, time_spec=TimeSpec("hjd"),
        earth_ephemeris=earth,
    )
    debug = evaluator.trajectory(np.asarray([1.0, 0.0, 0.1, 0.0, 0.0]), components=True)
    np.testing.assert_allclose(debug["observer_n"][1], 0.0, atol=1e-14)
    np.testing.assert_allclose(debug["observer_e"][1], 0.0, atol=1e-14)
    np.testing.assert_allclose(debug["tau"], [-1.0, 0.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(debug["beta"], 0.1, atol=1e-14)


def test_geocentric_satellite_offset_is_not_reference_subtracted():
    earth = Ephemeris(np.asarray([0.0, 1.0, 2.0]), np.zeros((3, 3)), origin="sun", time_spec=TimeSpec("hjd"))
    satellite = Ephemeris(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([[0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4]]),
        origin="earth", time_spec=TimeSpec("hjd"),
    )
    evaluator = ParallaxEvaluator(
        np.asarray([1.0]), np.ones(1), np.ones(1), ra_deg=0.0, dec_deg=0.0, tref=1.0,
        time_spec=TimeSpec("hjd"), earth_ephemeris=earth,
        satellite_or_observer_ephemeris=satellite,
    )
    raw = np.asarray([1.0, 0.0, 0.1, 0.0, 1.0])
    debug = evaluator.trajectory(raw, components=True)
    expected = [debug[key] for key in ("tau", "beta", "u")]
    np.testing.assert_array_equal(evaluator.trajectory(raw), expected)
    evaluator.magnification(raw)
    np.testing.assert_array_equal(evaluator.trajectory(raw), expected)
    np.testing.assert_allclose(debug["satellite_n"], [0.3], atol=1e-14)
    np.testing.assert_allclose(debug["beta"], [0.4], atol=1e-14)


def test_strict_gulls_requires_explicit_reference_and_keeps_reference_zero():
    observer = _linear_ephemeris(origin="explicit_reference", time_spec=TimeSpec("jd"))
    reference = Ephemeris(
        observer.time,
        observer.position_au * 0.5,
        origin="explicit_reference", time_spec=TimeSpec("jd"),
    )
    evaluator = ParallaxEvaluator(
        np.asarray([1.0]), np.ones(1), np.ones(1), ra_deg=0.0, dec_deg=0.0, tref=1.0,
        time_spec=TimeSpec("jd"), observer_convention="gulls",
        satellite_or_observer_ephemeris=observer, reference_ephemeris=reference,
    )
    debug = evaluator.trajectory(np.asarray([1.0, 0.0, 0.1, 0.0, 1.0]), components=True)
    np.testing.assert_allclose(debug["observer_n"], [0.1], atol=1e-14)
    with pytest.raises(ValueError, match="reference"):
        ParallaxEvaluator(
            np.asarray([1.0]), np.ones(1), np.ones(1), ra_deg=0.0, dec_deg=0.0, tref=1.0,
            time_spec=TimeSpec("jd"), observer_convention="gulls",
            satellite_or_observer_ephemeris=observer,
        )


def test_origin_metadata_is_not_guessed_from_distance():
    earth_like = Ephemeris(np.asarray([0.0, 1.0]), np.zeros((2, 3)), origin="earth")
    with pytest.raises(ValueError, match="origin"):
        ParallaxEvaluator(
            np.asarray([0.0, 1.0]), np.ones(2), np.ones(2), ra_deg=0.0, dec_deg=0.0, tref=0.5,
            time_spec=TimeSpec("hjd"), earth_ephemeris=earth_like,
        )


def test_time_spec_is_required_in_public_evaluator():
    with pytest.raises(ValueError, match="TimeSpec"):
        ParallaxEvaluator(
            np.asarray([0.0, 1.0]), np.ones(2), np.ones(2),
            ra_deg=0.0, dec_deg=0.0, tref=0.5,
            earth_ephemeris=Ephemeris(np.asarray([0.0, 1.0]), np.zeros((2, 3)), origin="sun"),
        )


def test_native_module_has_no_jax_import():
    from pathlib import Path
    source = Path(__file__).parents[1] / "src" / "singlelensfitter" / "parallax_backend.py"
    assert "import jax" not in source.read_text(encoding="utf-8")


def test_native_continuation_seed_does_not_apply_log_twice():
    time_spec = TimeSpec("hjd", offset=2450000.0)
    earth = Ephemeris(
        np.asarray([2458990.0, 2459000.0, 2459010.0]),
        np.zeros((3, 3)),
        origin="sun",
        time_spec=time_spec,
    )
    fitter = PSPLParallaxFitter(
        0.0, 0.0, 9000.0, time_spec=time_spec, earth_ephemeris=earth
    )
    raw = fitter._raw_seed(np.asarray([9000.0, 80.0, 0.1, 0.2, -0.1]))
    np.testing.assert_allclose(
        raw,
        [2459000.0, np.log(80.0), 0.1, 0.2, -0.1],
        atol=1.0e-14,
    )


def test_native_annual_fit_recovers_synthetic_and_removes_effect_score():
    time = np.linspace(0.0, 20.0, 201)
    ephemeris_time = np.linspace(-5.0, 25.0, 61)
    earth = Ephemeris(
        ephemeris_time,
        np.column_stack(
            [
                np.zeros_like(ephemeris_time),
                np.zeros_like(ephemeris_time),
                0.002 * (ephemeris_time - 10.0) ** 2,
            ]
        ),
        origin="sun",
        time_spec=TimeSpec("hjd"),
    )
    truth_raw = np.asarray([10.0, np.log(4.0), 0.12, 0.25, -0.15])
    geometry = ParallaxEvaluator(
        time,
        np.ones_like(time),
        np.ones_like(time),
        ra_deg=0.0,
        dec_deg=0.0,
        tref=10.0,
        time_spec=TimeSpec("hjd"),
        earth_ephemeris=earth,
    )
    magnification = geometry.magnification(truth_raw)
    flux = 1.7 * magnification + 0.3
    ferr = np.full_like(time, 0.01)
    fitter = PSPLParallaxFitter(
        0.0,
        0.0,
        10.0,
        time_spec=TimeSpec("hjd"),
        earth_ephemeris=earth,
        maxiter=500,
        tol=1.0e-10,
    )
    fit = fitter.fit(time, flux, ferr, [10.2, 3.7, 0.14, 0.1, -0.05])
    auto_fit = fitter.fit(time, flux, ferr)
    np.testing.assert_allclose(np.asarray(auto_fit.params), np.asarray(fit.params), rtol=1.0e-4, atol=1.0e-4)

    np.testing.assert_allclose(
        np.asarray(fit.params),
        [10.0, 4.0, 0.12, 0.25, -0.15],
        atol=1.0e-7,
    )
    assert float(fit.chi2) < 1.0e-12
    assert fit.diagnostics.backend == "scipy_trf_bounded_compiled_evaluator"
    assert parallax_effect_score(fit) < 1.0e-12


def test_parallax_tE_lower_bound_is_independent_of_survey_span():
    fitter = PSPLParallaxFitter(
        0.0, 0.0, 1.0, time_spec=TimeSpec("hjd"),
        earth_ephemeris=_linear_ephemeris(time_spec=TimeSpec("hjd")),
    )
    lower, _ = fitter._bounds(np.asarray([-856.0, 856.0]))

    np.testing.assert_allclose(np.exp(lower[1]), 1.0e-6, rtol=0.0, atol=1.0e-18)


def test_constant_magnification_uses_finite_constant_flux_profile():
    magnification = np.ones(5)
    flux = np.asarray([1.1, 0.9, 1.2, 1.0, 0.8])
    ferr = np.asarray([0.1, 0.2, 0.1, 0.3, 0.2])
    weights = 1.0 / ferr**2
    expected_mean = float(np.sum(weights * flux) / np.sum(weights))

    fs, fb = _profile_fluxes(magnification, flux, ferr)

    assert fs == 0.0
    np.testing.assert_allclose(fb, expected_mean)


def test_native_constant_magnification_evaluator_returns_constant_flux():
    time = np.linspace(-2.0, 2.0, 5)
    earth = Ephemeris(
        np.linspace(-3.0, 3.0, 7),
        np.zeros((7, 3)),
        origin="sun",
        time_spec=TimeSpec("hjd"),
    )
    flux = np.asarray([1.1, 0.9, 1.2, 1.0, 0.8])
    ferr = np.asarray([0.1, 0.2, 0.1, 0.3, 0.2])
    evaluator = ParallaxEvaluator(
        time,
        flux,
        ferr,
        ra_deg=0.0,
        dec_deg=0.0,
        tref=0.0,
        time_spec=TimeSpec("hjd"),
        earth_ephemeris=earth,
    )
    raw = np.asarray([1.0e6, np.log(0.1), 0.1, 0.0, 0.0])
    expected_mean = float(np.sum((flux / ferr**2)) / np.sum(1.0 / ferr**2))

    residual = evaluator.residual(raw)
    model = evaluator.evaluate(raw)

    assert np.all(np.isfinite(residual))
    np.testing.assert_allclose(model, expected_mean, rtol=0.0, atol=1.0e-14)


def test_native_fixed_evaluation_preserves_public_seed_coordinates():
    time = np.linspace(0.0, 2.0, 21)
    earth = Ephemeris(
        np.asarray([-1.0, 0.0, 1.0, 2.0, 3.0]),
        np.zeros((5, 3)),
        origin="sun",
        time_spec=TimeSpec("hjd"),
    )
    fitter = PSPLParallaxFitter(
        0.0, 0.0, 1.0, time_spec=TimeSpec("hjd"), earth_ephemeris=earth
    )
    seed = np.asarray([1.0, 0.4, 0.1, 0.2, -0.3])
    geometry = ParallaxEvaluator(
        time, np.ones_like(time), np.ones_like(time),
        ra_deg=0.0, dec_deg=0.0, tref=1.0, time_spec=TimeSpec("hjd"),
        earth_ephemeris=earth,
    )
    magnification = geometry.magnification(
        np.asarray([seed[0], np.log(seed[1]), *seed[2:]])
    )
    flux = 2.0 * magnification + 0.5
    fit = fitter.evaluate_fixed(time, flux, np.full_like(time, 0.01), seed)

    np.testing.assert_allclose(np.asarray(fit.params), seed, atol=1.0e-12)
    assert float(fit.chi2) < 1.0e-20
    assert fit.optimizer_status == "fixed_parameters"




@pytest.mark.parametrize("finite_source", [False, True])
@pytest.mark.parametrize("scale", ["jd", "hjd"])
def test_analytic_jacobian_matches_finite_difference(finite_source, scale):
    spec = TimeSpec(scale)
    ephemeris_time = np.linspace(-5.0, 25.0, 61)
    earth = Ephemeris(
        ephemeris_time,
        np.column_stack([
            0.01 * (ephemeris_time - 10.0) ** 2,
            0.3 * np.sin(ephemeris_time / 5.0),
            0.002 * (ephemeris_time - 10.0) ** 2,
        ]),
        origin="sun", time_spec=spec,
    )
    time = np.linspace(0.0, 20.0, 400)
    dataset = np.arange(time.size) % 2
    kwargs = dict(
        ra_deg=268.0, dec_deg=-29.0, tref=10.0, time_spec=spec,
        earth_ephemeris=earth, finite_source=finite_source,
    )
    geometry = ParallaxEvaluator(time, np.ones_like(time), np.ones_like(time), **kwargs)
    if finite_source:
        raw = np.asarray([10.3, np.log(4.0), 0.05, np.log(0.08), 0.25, -0.15])
    else:
        raw = np.asarray([10.3, np.log(4.0), 0.12, 0.25, -0.15])
    magnification = geometry.magnification(raw)
    rng = np.random.default_rng(3)
    flux = np.where(dataset == 0, 2.0, 0.7) * magnification + 0.3 + 0.01 * rng.normal(size=time.size)
    evaluator = ParallaxEvaluator(
        time, flux, np.full_like(time, 0.01), dataset, **kwargs
    )
    mask = np.ones(time.size, dtype=bool)
    mask[50:80] = False
    for active in (None, mask):
        native = evaluator._native
        fd = np.asarray(native.jacobian(raw, active_mask=active, method="fd"))
        analytic = np.asarray(native.jacobian(raw, active_mask=active))
        scale_per_column = np.linalg.norm(fd, axis=0)
        error = np.linalg.norm(analytic - fd, axis=0) / scale_per_column
        assert np.all(error < 2.0e-4), error
    # The same parameter vector may be reused after an in-place mask change.
    mask[:] = dataset == 0
    shifted = raw.copy()
    shifted[0] += .2
    evaluator.residual(shifted, mask)
    residual = evaluator.residual(raw, mask)
    jacobian = evaluator.jacobian(raw, mask)
    joint_res, joint_jac = evaluator.residual_and_jacobian(raw, mask)
    np.testing.assert_array_equal(joint_res, residual)
    np.testing.assert_array_equal(joint_jac, jacobian)
    fresh = ParallaxEvaluator(time, flux, np.full_like(time, .01), dataset, **kwargs)
    np.testing.assert_array_equal(residual, fresh.residual(raw, mask))
    np.testing.assert_array_equal(jacobian, fresh.jacobian(raw, mask))
    np.testing.assert_array_equal(evaluator.evaluate(raw, mask), fresh.evaluate(raw))
    with pytest.raises(ValueError):
        evaluator._native.jacobian(raw, method="nope")


@pytest.mark.parametrize("kind", ["pspl", "fspl"])
def test_single_lens_analytic_jacobian_matches_finite_difference(kind):
    from scipy.optimize._numdiff import approx_derivative

    from singlelensfitter import fitters

    rng = np.random.default_rng(3)
    time = np.linspace(-40.0, 40.0, 300)
    ferr = np.full(time.size, 0.02)
    z = np.array([1.0, np.log(12.0), 0.05, np.log(0.08)] if kind == "fspl" else [1.0, np.log(12.0), 0.3])
    table = fitters.default_espl_table_path()

    def magnification_and_gradient(z):
        if kind == "pspl":
            q = fitters.PSPLFitter._decode(z)
            return fitters._pspl_magnification(time, *q), fitters._pspl_gradient(time, *q)
        q = fitters.FSPLFitter._decode(z)
        u, du = fitters._trajectory_gradient(time, *q[:3])
        rho = np.exp(q[3])
        a, d_du, d_drho = fitters._cpp_fspl_magnification_gradient(
            u, rho, espl_table_path=table, tol=1e-4, reltol=1e-4
        )
        return a, np.column_stack([d_du[:, None] * du, d_drho * rho])

    flux = magnification_and_gradient(z)[0] + 0.3 + rng.normal(0.0, 0.02, time.size)

    def residual(z):
        a = magnification_and_gradient(z)[0]
        fs, fb = fitters._solve_fs_fb_numpy(a, flux, ferr)
        return (flux - fs * a - fb) / ferr

    a, gradient = magnification_and_gradient(z)
    analytic = fitters._profiled_residual_jacobian(a, gradient, flux, ferr)
    numeric = approx_derivative(residual, z, method="2-point", rel_step=1e-7)
    error = np.abs(analytic - numeric).max(axis=0) / np.abs(numeric).max(axis=0)
    assert np.all(error < 1e-5)


@pytest.mark.parametrize("offset", [0.0, 2460000.0])
def test_annual_fit_recovers_short_event_on_absolute_and_reduced_time_axes(offset):
    origin = 2463106.48
    dates = origin + np.linspace(-0.4, 0.4, 401)
    eph_dates = origin + np.linspace(-1.0, 1.0, 101)
    spec = TimeSpec("hjd", offset=offset)
    pos = np.zeros((len(eph_dates), 3))
    pos[:, 2] = 0.1 * (eph_dates - origin)**2
    earth = Ephemeris(eph_dates, pos, origin="sun", time_spec=spec)
    # At zero parallax the trajectory is analytic even at full Julian dates.
    u = np.hypot((dates - origin) / 0.06, 0.376)
    flux = 1.7 * (u*u + 2) / (u * np.sqrt(u*u + 4)) + 0.2
    fitter = PSPLParallaxFitter(0, 0, origin - offset, time_spec=spec, earth_ephemeris=earth)
    fit = fitter.fit(dates - offset, flux, np.full_like(flux, 0.01),
                     [origin - offset - 0.01, 0.0675, 0.05, 0, 0])
    assert fit.optimizer_success
    assert float(fit.chi2) < 1.0e-8


def test_default_earth_ephemerides_have_independent_arrays():
    from singlelensfitter.parallax_backend import default_earth_ephemeris

    first = default_earth_ephemeris()
    original = first.position_au[0, 0]
    first.position_au[0, 0] = original + 1.
    second = default_earth_ephemeris(time_spec=TimeSpec("hjd"))
    assert second.position_au[0, 0] == original
    assert second.time_spec == TimeSpec("hjd")
