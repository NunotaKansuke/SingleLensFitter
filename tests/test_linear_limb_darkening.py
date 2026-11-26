import runpy
from pathlib import Path

import numpy as np
import pytest

from singlelensfitter import FSPLFitter, FSPLParallaxFitter, fspl_magnification_gradient
from singlelensfitter.parallax_backend import Ephemeris, ParallaxEvaluator, TimeSpec

# Independent lens-centered direct disk quadrature, including radial integration.
reference = runpy.run_path(str(Path(__file__).with_name("limb_reference.py")))["magnification"]


@pytest.mark.parametrize("a", [0., 0.3, 1.])
def test_physical_partials_against_brightness_integral(a):
    r = 0.72
    u = r*np.array([0.05, 0.55, 0.99, 1.001, 1.035])
    value, du, dr, da = fspl_magnification_gradient(u, r, a)
    ref = lambda u, r, a: reference(u, r, a, 64)
    np.testing.assert_allclose(value, ref(u, r, a), rtol=1e-8)
    for h in (1e-6, 3e-7):
        np.testing.assert_allclose(du, (ref(u+h,r,a)-ref(u-h,r,a))/(2*h), rtol=2e-5)
        np.testing.assert_allclose(dr, (ref(u,r+h,a)-ref(u,r-h,a))/(2*h), rtol=2e-5)
        np.testing.assert_allclose(da, (ref(u,r,a+h)-ref(u,r,a-h))/(2*h), rtol=2e-5)


def test_pure_mu_contact_and_far_field():
    r = 0.35
    # Pure mu vanishes at the limb. Its first derivatives remain finite.
    u = r*np.array([1., 1.201, 3.9, 4.1, 8., 270.])
    value, du, dr, _ = fspl_magnification_gradient(u,r,1.)
    np.testing.assert_allclose(value, reference(u,r,1.,64), rtol=1e-8)
    h = 1e-6
    # At contact the derivative of the pure-mu law is continuous but has
    # a square-root cusp. Extrapolate centered slopes in sqrt(step).
    for col,delta_u,delta_r in ((du,1,0),(dr,0,1)):
        def slope(h):
            return (reference(u+h*delta_u,r+h*delta_r,1.,80)-
                    reference(u-h*delta_u,r-h*delta_r,1.,80))/(2*h)
        np.testing.assert_allclose(col, 2*slope(h/4)-slope(h), rtol=1e-3, atol=2e-7)


def test_recover_free_coefficient_and_replay():
    t = np.linspace(-1.8,1.8,151)
    truth = [0.012, 0.9, 0.11, 0.65, 0.55]
    t0, te, u0, rho, a = truth
    flux = 2*reference(np.hypot((t-t0)/te,u0),rho,a,64)+0.3
    err = np.full(t.size, 0.01)
    fitter = FSPLFitter(fit_limb_darkening=True, tol=1e-10)
    fit = fitter.fit(t,flux,err,[0.,1.,0.12,np.log(.6),.4])
    assert fit.optimizer_success
    assert fit.param_names[-1] == "a"
    measured = np.asarray(fit.params).copy()
    measured[2] = abs(measured[2])  # Rectilinear photometry is symmetric in u0.
    np.testing.assert_allclose(measured, truth, atol=0.002)
    replay = fitter.evaluate_fixed(t,flux,err,fitter.seed_from_fit(fit))
    np.testing.assert_allclose(replay.model_flux,fit.model_flux)
    assert fit.linear_limb_darkening == float(fit.params[-1])


def test_parallax_profile_jacobian_with_fixed_limb_darkening():
    spec = TimeSpec("hjd")
    t = np.linspace(-1,1,48)
    earth = Ephemeris(t, np.zeros((len(t),3)), origin="sun",time_spec=spec)
    ev = ParallaxEvaluator(t,np.cos(t),np.full_like(t,.05),ra_deg=0,dec_deg=0,tref=0,
                           time_spec=spec,earth_ephemeris=earth,finite_source=True,
                           linear_limb_darkening=.6)
    p = np.array([0.,np.log(2.),.2,np.log(.7),0,0])
    for j in range(4):
        d = np.zeros(6)
        d[j] = 1e-6
        fd = (ev.residual(p+d)-ev.residual(p-d))/(2e-6)
        np.testing.assert_allclose(ev.jacobian(p)[:,j],fd,rtol=4e-5,atol=2e-6)
    magnification = fspl_magnification_gradient(np.hypot(t/2,.2),.7,.6)[0]
    np.testing.assert_allclose(ev.magnification_at(t,p), magnification)
    fitter = FSPLParallaxFitter(0, 0, 0, time_spec=spec, earth_ephemeris=earth,
                                linear_limb_darkening=.6)
    seed = p.copy()
    seed[1] = 2
    flux = .1 + 2*magnification
    fit = fitter.fit(t, flux, np.full_like(t,.05), seed)
    assert fit.linear_limb_darkening == .6
    assert float(fit.chi2) < 1e-15
    fixed = fitter.evaluate_fixed(t, flux, np.full_like(t,.05), fitter.seed_from_fit(fit))
    np.testing.assert_allclose(fixed.model_flux, fit.model_flux)



def test_joint_fit_passes_profiled_five_column_jacobian(monkeypatch):
    from singlelensfitter import fitters

    original = fitters.least_squares
    checked = []

    def check_and_fit(fun, start, *, jac, **kwargs):
        analytic = jac(start)
        for j in range(5):
            delta = np.zeros(5)
            delta[j] = 1e-6
            fd = (fun(start+delta)-fun(start-delta))/(2e-6)
            np.testing.assert_allclose(analytic[:,j], fd, rtol=1e-4, atol=1e-5)
        checked.append(True)
        return original(fun, start, jac=jac, **kwargs)

    monkeypatch.setattr(fitters, "least_squares", check_and_fit)
    t = np.linspace(-0.7,0.7,51)
    flux = reference(np.hypot(t,.12),.8,.6)
    FSPLFitter(fit_limb_darkening=True).fit(t, flux, np.full_like(t,.02),
                                            [0., 1., .13, np.log(.83), .5])
    assert checked


@pytest.mark.parametrize("a", [-0.1, np.nan, 1.01])
def test_invalid_linear_coefficient(a):
    with pytest.raises(ValueError):
        FSPLFitter(linear_limb_darkening=a)
    with pytest.raises(ValueError):
        fspl_magnification_gradient([0.1], 1., a)


def test_large_source_mu_basis():
    rho = 100.
    u = rho*np.array([.1, .5, .99, 1., 1.01, 1.5])
    A, du, dr, _ = fspl_magnification_gradient(u, rho, 1.)
    np.testing.assert_allclose(A, reference(u, rho, 1., 96), rtol=1e-8)
    use = u != rho
    h = 1e-4
    for derivative, dx, ds in ((du, 1, 0), (dr, 0, 1)):
        expected = (reference(u[use]+dx*h, rho+ds*h, 1., 96) -
                    reference(u[use]-dx*h, rho-ds*h, 1., 96))/(2*h)
        np.testing.assert_allclose(derivative[use], expected, rtol=2e-4, atol=1e-10)


@pytest.mark.parametrize("rho", [.01, .125, 1., 1.95, 2.])
def test_mu_table_gradients_across_patches_and_blends(rho):
    z = np.array([0., .34, np.sqrt(.5), .995, 1.005, 1.25, 2., 3.25, 5., 9.4, 10.])
    u = rho*z
    value, du, dr, _ = fspl_magnification_gradient(u, rho, 1.)
    np.testing.assert_allclose(value, reference(u, rho, 1., 96), rtol=1e-8)
    step = rho*1e-5
    for result, dx, ds in ((du, 1, 0), (dr, 0, 1)):
        upper = reference(abs(u+dx*step), rho+ds*step, 1., 96)
        lower = reference(abs(u-dx*step), rho-ds*step, 1., 96)
        np.testing.assert_allclose(result*rho**2, (upper-lower)/(2*step)*rho**2,
                                   atol=1e-8, rtol=2e-5)


@pytest.mark.parametrize("rho", [.01, 1.98])
def test_mu_contact_blend_derivative(rho):
    # The second radius exercises overlapping contact and radius transitions.
    z = 1 + 1e-6*np.array([-2., -1.5, -1., 1., 1.5, 2.])
    u = rho*z
    _, du, dr, _ = fspl_magnification_gradient(u, rho, 1.)
    for step in (rho*1e-8, rho*3e-9):
        for result, dx, ds in ((du, 1, 0), (dr, 0, 1)):
            upper = reference(u+dx*step, rho+ds*step, 1., 96)
            lower = reference(u-dx*step, rho-ds*step, 1., 96)
            np.testing.assert_allclose(result, (upper-lower)/(2*step), rtol=2e-5)


def test_uniform_limb_table_with_changing_radius():
    # Cross each logarithmic cell boundary in both directions and reuse a radius.
    gap = np.array([2.**-16, 1.5*2.**-16, 2.**-15, 2.**-12, 2.**-8, .01, .05])
    z = 1 + np.r_[-gap, gap]
    for rho in (.01, .5, 1.95, .01):
        u = rho*z
        value, du, dr, _ = fspl_magnification_gradient(u, rho, 0.)
        np.testing.assert_allclose(value, reference(u,rho,0.,96), rtol=1e-9)
        step = rho*1e-8
        for result, dx, ds in ((du, 1, 0), (dr, 0, 1)):
            upper = reference(u+dx*step, rho+ds*step, 0., 96)
            lower = reference(u-dx*step, rho-ds*step, 0., 96)
            np.testing.assert_allclose(result, (upper-lower)/(2*step), rtol=2e-5)


@pytest.mark.parametrize("rho", [.001, .35, 1.95, 10., 100.])
def test_exterior_mu_series_against_disk_integrals(rho):
    # Differentiate the point kernel at each point on the source disk.
    # This also resolves dA/drho in the wings where differencing A loses digits.
    x, w = np.polynomial.legendre.leggauss(40)
    mu = (1+x[:, None])/2
    radius = np.sqrt(1-mu**2)
    cosine = np.cos(np.pi*(1+x)/2)
    weight = .75*mu**2*w[:, None]*w
    u = rho*np.array([8.9999999, 9., 9.5, 10., 12., 50., 256., 1500.])
    value, du, dr, _ = fspl_magnification_gradient(u, rho, 1.)
    for i, separation in enumerate(u):
        d = np.hypot(separation-rho*radius*cosine, rho*radius*np.sqrt(1-cosine**2))
        h = np.hypot(d, 2.)
        slope_over_d = -8/(d**3*h**3)
        expected = [np.sum(weight*(d/h+2/(d*h))),
                    np.sum(weight*slope_over_d*(separation-rho*radius*cosine)),
                    np.sum(weight*slope_over_d*(rho*radius**2-separation*radius*cosine))]
        np.testing.assert_allclose([value[i], du[i], dr[i]], expected, rtol=2e-7)


def test_mixed_wing_runs_and_coefficient_derivative():
    rho, a = .25, .45
    # Alternate near-field samples with wing runs of different lengths.
    u = rho*np.array([200., 1., 2., 64., 64.0001, 99., 1000., .98, 90.])
    value, du, dr, da = fspl_magnification_gradient(u, rho, a)
    for j, separation in enumerate(u):
        single = np.asarray(fspl_magnification_gradient([separation], rho, a)).ravel()
        np.testing.assert_allclose([value[j],du[j],dr[j],da[j]], single, rtol=1e-12)
    # The uniform basis uses the VBM quadrupole here. Combine it with an
    # independent disk integral for mu, using excess flux to retain precision.
    x, w = np.polynomial.legendre.leggauss(48)
    mu = (1+x[:,None])/2
    radius = np.sqrt(1-mu**2)
    cosine = np.cos(np.pi*(1+x)/2)
    weight = .75*mu**2*w[:,None]*w
    for j in np.flatnonzero(u > 64*rho):
        sep = u[j]
        h = np.hypot(sep, 2.)
        point = 4/(sep*h*(sep*sep+2+sep*h))
        quad = 4*(sep*sep+1)*rho*rho/(sep**3*h**5)
        d = np.sqrt(sep*sep+rho**2*radius**2-2*sep*rho*radius*cosine)
        hd = np.hypot(d, 2.)
        excess = 4/(d*hd*(d*d+2+d*hd))
        difference = np.sum(weight*(excess-point)) - quad
        np.testing.assert_allclose(da[j], 6*difference/(3-a)**2, rtol=2e-8)
