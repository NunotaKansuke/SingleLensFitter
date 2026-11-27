import numpy as np
import pytest

from singlelensfitter import (
    Ephemeris, FSPLFitter, FSPLParallaxFitter, PSPLFitter, PSPLParallaxFitter, TimeSpec,
)


@pytest.mark.parametrize("kind", ["point", "finite", "limb", "annual", "space"])
def test_prepared_mask_gradient_and_continuation(kind):
    t = np.linspace(-3., 3., 181)
    flux = 2. + 3. / np.sqrt(.2**2 + t*t)
    error = .2 + .01*t
    mask = np.arange(t.size) % 9 == 0
    flux[mask] += 20
    q = [.035, .9, .17]
    if kind in ("finite", "limb", "space"):
        q.append(.37)
    if kind in ("annual", "space"):
        spec = TimeSpec("hjd")
        ep = np.linspace(-10., 10., 101)
        pos = np.column_stack((0*ep, ep*ep*.02, ep*ep*.01))
        opts = dict(time_spec=spec, earth_ephemeris=Ephemeris(ep, pos, origin="sun", time_spec=spec))
        if kind == "space":
            opts.update(geometry="space", satellite_or_observer_ephemeris=Ephemeris(
                ep, pos*.1, origin="earth", time_spec=spec))
        cls = FSPLParallaxFitter if kind == "space" else PSPLParallaxFitter
        fitter = cls(35, -20, 0., **opts)
        q.extend([.1, -.05])
    elif kind == "point":
        fitter = PSPLFitter()
    else:
        fitter = FSPLFitter(fit_limb_darkening=kind == "limb")
        if kind == "limb":
            q.append(.3)
    q = np.array(q)
    curve = fitter.prepare(t, flux, error)
    # Independent weighting and flux regression at each displaced lens solution.
    def magnitude(values):
        seed = np.asarray(values).copy()
        if kind in ("finite", "limb", "space"):
            seed[3] = np.log(seed[3])
        if kind in ("space", "annual"):
            return fitter._evaluator(t, flux, error).magnification(fitter._raw_seed(seed))
        return fitter._magnification(t, seed)

    def objective(params):
        design = np.column_stack((magnitude(params), np.ones_like(t)))[~mask] / error[~mask,None]
        y = flux[~mask] / error[~mask]
        r = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        return r @ r

    chi, grad = curve.chi2_and_grad(q, mask=mask)
    assert chi == pytest.approx(objective(q), rel=1e-9)
    for step in (2e-6, 6e-7):
        fd = [(objective(q+d)-objective(q-d))/(2*step) for d in np.eye(len(q))*step]
        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=2e-4)
    assert curve.chi2_and_grad(q)[0] > chi
    assert curve.chi2_and_grad(q, mask=mask)[0] == chi
    fit = curve.fit(start=q, mask=mask)
    np.testing.assert_array_equal(fit.mask, mask)
    np.testing.assert_allclose(fit.model_flux, float(fit.fs)*magnitude(fit.params)+float(fit.fb),
                               rtol=1e-9)
    np.testing.assert_allclose(fit.chi2, np.sum((fit.residual[~mask]/error[~mask])**2), rtol=1e-8)
    assert curve.fit(start=fit, mask=mask).chi2 <= fit.chi2*(1+1e-7)


def test_prepared_snapshots_inputs_and_validates_masks():
    t = np.linspace(-2, 2, 81)
    f = np.exp(-t*t)
    curve = PSPLFitter().prepare(t, f, np.ones_like(f))
    q = [0., .8, .24]
    chi = curve.chi2_and_grad(q)[0]
    f[:] = 0
    assert curve.chi2_and_grad(q)[0] == chi
    for mask in (np.ones(t.size, dtype=bool), np.ones(3, dtype=bool), np.ones(t.size)):
        with pytest.raises(ValueError):
            curve.fit(mask=mask)
        with pytest.raises(ValueError):
            curve.chi2_and_grad(q, mask=mask)


def test_masked_auto_fit_ignores_excluded_flux_and_is_unit_invariant():
    t = np.linspace(-2, 2, 201)
    u = np.hypot((t-.2)/.7, .33)
    f = 2.2*(u*u+2)/(u*np.hypot(u, 2)) + .1 + .002*np.sin(20*t)
    mask = abs(t+.9) < .1
    f[mask] += 12
    err = .05 + (t+2)*.01
    fitter = PSPLFitter()
    result = fitter.prepare(t, f, err).fit(mask=mask)
    subset = fitter.fit(t[~mask], f[~mask], err[~mask])
    np.testing.assert_allclose(result.params, subset.params)
    assert result.chi2 == pytest.approx(subset.chi2)
    scaled = fitter.prepare(t, 1e-12*f, 1e-12*err)
    refit = scaled.fit(start=result, mask=mask)
    chi, _ = scaled.chi2_and_grad(refit.params, mask=mask)
    assert chi == pytest.approx(result.chi2)
    assert float(refit.chi2) == pytest.approx(chi)
