from __future__ import annotations

import numpy as np
import pytest




def test_compiled_fspl_magnification_binding_is_available():
    native = pytest.importorskip("singlelensfitter._vbm_cpp")
    from singlelensfitter.parallax_backend import default_espl_table_path

    table_path = default_espl_table_path()
    if table_path is None:
        pytest.skip("The compiled finite-source table is not available.")
    values = np.asarray(
        native.fspl_magnification(
            np.asarray([0.01, 0.1, 1.0, 3.0]),
            0.05,
            espl_table=table_path,
        ),
        dtype=float,
    )

    assert values.shape == (4,)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 1.0)


def test_canonical_pspl_fitter_uses_scipy_lm():
    from singlelensfitter import PSPLFitter

    time = np.linspace(-15.0, 15.0, 121)
    truth = np.asarray([1.5, 4.0, 0.2])
    u = np.sqrt(((time - truth[0]) / truth[1]) ** 2 + truth[2] ** 2)
    magnification = (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))
    flux = 1.7 * magnification + 0.2
    fit = PSPLFitter(maxiter=500, tol=1.0e-10).fit(
        time, flux, np.full_like(time, 0.01), [1.0, 3.5, 0.25]
    )

    assert fit.model_kind == "pspl"
    assert fit.optimizer_success
    assert fit.optimizer_status.startswith("scipy_lm:")
    np.testing.assert_allclose(np.asarray(fit.params), truth, atol=1.0e-7)
    assert float(fit.chi2) < 1.0e-12


def test_canonical_fspl_fitter_uses_compiled_magnification():
    from singlelensfitter import FSPLFitter

    time = np.linspace(90.0, 110.0, 201)
    truth = np.asarray([100.0, 4.0, 0.15, 0.08])
    fitter = FSPLFitter(maxiter=500, tol=1.0e-10)
    magnification = fitter._magnification(
        time,
        np.asarray([truth[0], truth[1], truth[2], np.log(truth[3])]),
    )
    flux = 1.7 * magnification + 0.2
    fit = fitter.fit(
        time, flux, np.full_like(time, 0.01), [100.2, 3.5, 0.18, np.log(0.1)]
    )

    assert fit.model_kind == "fspl"
    assert fit.optimizer_success
    assert "compiled_magnification" in fit.optimizer_status
    np.testing.assert_allclose(np.asarray(fit.params), truth, atol=1.0e-6)
    assert float(fit.chi2) < 1.0e-10


def test_fspl_fit_is_stable_for_large_absolute_event_times():
    from singlelensfitter import FSPLFitter

    # This reproduces the scale separation in the raw RMDC data: a short event
    # is represented on a full BJD axis around 2,463,106.
    truth = np.asarray([2463106.48, 0.06, 0.376, 1.02])
    time = np.linspace(truth[0] - 0.4, truth[0] + 0.4, 401)
    fitter = FSPLFitter(maxiter=500, tol=1.0e-10)
    q_truth_local = np.asarray([0.0, truth[1], truth[2], np.log(truth[3])])
    magnification = fitter._magnification(time - truth[0], q_truth_local)
    flux = 1.7 * magnification + 0.2
    fit = fitter.fit(
        time,
        flux,
        np.full_like(time, 0.01),
        [truth[0] - 0.01, 0.0675, 0.05, 0.0],
    )

    assert fit.optimizer_success
    fitted = np.asarray(fit.params)
    np.testing.assert_allclose(fitted[[0, 1, 3]], truth[[0, 1, 3]], atol=1.0e-5)
    np.testing.assert_allclose(abs(fitted[2]), truth[2], atol=1.0e-5)
    assert float(fit.chi2) < 1.0e-8


def test_pspl_solution_is_invariant_to_absolute_bjd():
    from singlelensfitter import PSPLFitter

    offset = 2463106.48
    absolute = offset + np.linspace(-0.4, 0.4, 401)
    relative = absolute - offset
    u = np.hypot(relative / 0.06, 0.376)
    flux = 1.7 * (u**2 + 2) / (u * np.sqrt(u**2 + 4)) + 0.2
    for time, origin in ((absolute, offset), (relative, 0.0)):
        fit = PSPLFitter().fit(time, flux, np.full_like(flux, 0.01),
                               [origin - 0.01, 0.0675, 0.05])
        assert fit.optimizer_success
        assert float(fit.chi2) < 1.0e-8
        np.testing.assert_allclose(np.asarray(fit.params) - [origin, 0, 0],
                                   [0, 0.06, 0.376], rtol=0, atol=1.0e-7)


@pytest.mark.parametrize("rho", [0.003, 0.35, 1.03, 3.0, 100.0])
def test_fspl_across_source_limb_against_disk_integral(rho):
    from scipy.integrate import quad
    from singlelensfitter.magnification import _native_fspl_magnification
    from singlelensfitter import _vbm_cpp
    from singlelensfitter.parallax_backend import default_espl_table_path

    # Integrate in lens-centered polar coordinates. The radial primitive
    # of r*A_PSPL(r) is r*sqrt(r**2+4)/2, including at the lens itself.
    def disk_average(u, rho=rho):
        limit = np.pi if u < rho else np.arcsin(rho / u)

        def angular_integrand(angle):
            center = u * np.cos(angle)
            half = np.sqrt(max(0, rho**2 - (u * np.sin(angle))**2))
            outer, inner = center + half, max(0, center - half)
            return outer * np.hypot(outer, 2) - inner * np.hypot(inner, 2)

        return quad(angular_integrand, 0, limit, epsabs=1e-14 * rho, epsrel=1e-12)[0] / (np.pi * rho**2)

    ratios = np.array([0.96, 0.99, 0.9999, 1.0, 1.0001, 1.01, 1.04])
    u = rho * ratios
    expected = [disk_average(x) for x in u]
    compiled = _vbm_cpp.fspl_magnification(u, rho, espl_table=default_espl_table_path())
    fallback = _native_fspl_magnification(u, rho)
    np.testing.assert_allclose(compiled, expected, rtol=1e-8)
    np.testing.assert_allclose(fallback, expected, rtol=1e-8)
    at_contact = 2/np.pi * (1/rho + (1 + 1/rho**2)*np.arctan(rho))
    np.testing.assert_allclose(compiled[3], at_contact, rtol=1e-12)
    # Compare each partial to differences of the independent disk integral,
    # holding the other physical coordinate fixed on either side of the limb.
    for ratio in (0.5, 0.874, 0.99, 1.01):
        x = ratio * rho
        _, du, dr = _vbm_cpp.fspl_magnification_gradient(
            np.array([x]), rho, espl_table=default_espl_table_path())
        for h in (1e-5 * rho, 2e-6 * rho):
            du_ref = (disk_average(x+h) - disk_average(x-h))/(2*h)
            dr_ref = (disk_average(x, rho+h) - disk_average(x, rho-h))/(2*h)
            np.testing.assert_allclose(du, du_ref, rtol=2e-5, atol=1e-10)
            np.testing.assert_allclose(dr, dr_ref, rtol=2e-5, atol=1e-10)
    if rho in (0.35, 1.03):
        off_limb = rho * np.array([0.0, 0.5, 2.0])
        values = _vbm_cpp.fspl_magnification(off_limb, rho, espl_table=default_espl_table_path())
        np.testing.assert_allclose(values, [disk_average(x) for x in off_limb], rtol=1e-4)


@pytest.mark.parametrize("z", [0.8499, 0.8501, 0.89999, 0.90001, 0.95, 0.9999, 1.0001, 1.05, 1.10, 1.1499, 1.1501])
def test_limb_gradient_including_transition(z):
    from singlelensfitter import _vbm_cpp
    from singlelensfitter.parallax_backend import default_espl_table_path

    table = default_espl_table_path()
    rho = 1.03
    u = np.array([z*rho])
    value, du, dr = _vbm_cpp.fspl_magnification_gradient(u, rho, espl_table=table)

    def evaluate(x, r):
        return _vbm_cpp.fspl_magnification(x, r, espl_table=table)

    np.testing.assert_allclose(value, evaluate(u, rho), rtol=1e-14)
    for h in (1e-6, 2e-7):
        np.testing.assert_allclose(du, (evaluate(u+h,rho)-evaluate(u-h,rho))/(2*h), rtol=2e-5)
        np.testing.assert_allclose(dr, (evaluate(u,rho+h)-evaluate(u,rho-h))/(2*h), rtol=2e-5)


def test_fspl_rejects_radii_beyond_espl_table():
    from singlelensfitter import _vbm_cpp
    from singlelensfitter.parallax_backend import default_espl_table_path

    for evaluate in (_vbm_cpp.fspl_magnification, _vbm_cpp.fspl_magnification_gradient):
        with pytest.raises(ValueError, match="rho"):
            evaluate(np.array([1.0]), 101.0, espl_table=default_espl_table_path())
    seeds = np.array([[0, 1, 0.1, np.log(110.0)]])
    score = _vbm_cpp.score_fspl_seeds(np.arange(4.), np.ones(4), np.ones(4), seeds,
                                      espl_table=default_espl_table_path())
    assert np.isposinf(score[0])


def test_fspl_fixed_accepts_last_espl_radius():
    from singlelensfitter import FSPLFitter

    t = np.array([0, 50, 100, 150.])
    fit = FSPLFitter().evaluate_fixed(t, np.ones(4), np.ones(4), [0, 1, 1, np.log(100.)])
    assert float(fit.params[3]) == 100
    assert np.isfinite(float(fit.chi2))


def test_limb_seed_score_matches_fixed_fit():
    from singlelensfitter import FSPLFitter, _vbm_cpp
    from singlelensfitter.parallax_backend import default_espl_table_path

    t = np.linspace(-2, 2, 101)
    flux = 3 + np.exp(-t**2)
    ferr = 0.03 + 0.02 * abs(t)
    seed = np.array([0.02, 1.0, 0.1, np.log(1.03)])
    score, = _vbm_cpp.score_fspl_seeds(t, flux, ferr, seed[None, :],
                                        espl_table=default_espl_table_path())
    fixed = FSPLFitter().evaluate_fixed(t, flux, ferr, seed)
    np.testing.assert_allclose(score, fixed.chi2, rtol=1e-12)


def test_contact_gradient_uses_symmetric_normal_secant():
    from singlelensfitter import _vbm_cpp
    from singlelensfitter.parallax_backend import default_espl_table_path

    table = default_espl_table_path()
    rho = 1.03
    step = 1e-6
    _, du, dr = _vbm_cpp.fspl_magnification_gradient(np.array([rho]), rho, espl_table=table)
    pair = _vbm_cpp.fspl_magnification(rho*np.array([1-step, 1+step]), rho, espl_table=table)
    np.testing.assert_allclose(du, (pair[1]-pair[0])/(2*step*rho), rtol=1e-8)
    # On the tangent u=rho, dA/drho follows from the closed-form contact value.
    np.testing.assert_allclose(du + dr, -4*np.arctan(rho)/(np.pi*rho**3), rtol=1e-12)


def test_near_contact_normal_derivative():
    from singlelensfitter import _vbm_cpp
    from singlelensfitter.parallax_backend import default_espl_table_path

    rho = 1.03
    u = rho * (1 + np.array([-1e-8, 1e-8, -1e-12, 1e-12]))
    # For these binary64 inputs, integrate the normal derivative at 45
    # decimal digits, split at |z-1| and sqrt(|z-1|) in polar angle.
    expected = [-10.884494486266465, -10.884494380333237,
                -16.411353386178973, -16.411353386162850]
    _, du, _ = _vbm_cpp.fspl_magnification_gradient(u, rho, espl_table=default_espl_table_path())
    np.testing.assert_allclose(du, expected, rtol=1e-7)


def test_fspl_smooth_disk_value_agrees_with_gradient_evaluator():
    from singlelensfitter import _vbm_cpp
    from singlelensfitter.parallax_backend import default_espl_table_path

    table = default_espl_table_path()
    z = np.r_[np.linspace(0, 1.15, 251), 0.9375, 1.0625]
    for rho in (1e-10, 0.01, 1.95, 2., 10., 100.):
        for a in (0., .5):
            kwargs = dict(espl_table=table, linear_limb_darkening=a)
            values = _vbm_cpp.fspl_magnification(z*rho, rho, **kwargs)
            differentiated = _vbm_cpp.fspl_magnification_gradient(z*rho, rho, **kwargs)
            np.testing.assert_allclose(values, differentiated[0], rtol=1e-13, atol=0)


def test_native_light_curve_matches_separation_batch_when_coefficient_changes():
    from singlelensfitter import FSPLFitter, _vbm_cpp

    model = FSPLFitter()
    time = np.linspace(-2, 2, 301)[::-2]
    q = np.array([.03, -1.4, -.1, np.log(.19), .5])
    u = np.sqrt(((time-q[0])/abs(q[1]))**2 + q[2]**2)
    for q[4] in (.5, 0., 1., .2):
        reference = _vbm_cpp.fspl_magnification(
            u, np.exp(q[3]), model.espl_table_path, 1e-4, 1e-4, q[4])
        np.testing.assert_allclose(model._magnification(time, q), reference, rtol=1e-14)


def test_jax_native_fspl_value_and_gradient_follow_the_input_trajectory():
    import jax
    import jax.numpy as jnp
    from singlelensfitter import fspl_magnification_gradient, jax_fspl_magnification

    pos = np.array([[-.01+.003j, .0097j], [.016-.002j, .08+.01j]])
    radii = np.abs(pos)
    weights = np.array([[.2, .4], [1.5, -.3]])
    rho, a = .01, .45
    value, du, dr, da = fspl_magnification_gradient(radii, rho, a)
    def total(offset, radius, coefficient):
        return jnp.sum(weights*jax_fspl_magnification(pos+offset, radius, coefficient))
    objective, gradients = jax.jit(jax.value_and_grad(total, argnums=(0, 1, 2)))(0., rho, a)
    np.testing.assert_allclose(objective, np.sum(weights*value), rtol=2e-13)
    np.testing.assert_allclose(gradients, [np.sum(weights*du*pos.real/radii),
                                          np.sum(weights*dr), np.sum(weights*da)], rtol=2e-12)
