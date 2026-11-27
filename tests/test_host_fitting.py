"""Control-flow regression tests for repeated host fits."""
import numpy as np

from singlelensfitter import auto_init
from singlelensfitter.host_fitting import PSPLFitter
from singlelensfitter.init_config import InitConfig


def test_mask_and_continuation_reuse_only_matching_searches(monkeypatch):
    fitter = PSPLFitter(init_config=InitConfig(auto_init_u0_grid_n=2, auto_init_fft_tE_grid_n=3))
    time = np.linspace(-6, 6, 301)
    ferr = np.full_like(time, 0.01)
    flux = 2 + fitter.magnification_model(time)([0.1, 2.5, 0.2])
    curve = fitter.prepare(time, flux, ferr)
    excluded = time < -5
    scans = []
    original = auto_init.PSPLFFTScanner.search_tE

    def count(scanner, epochs, *args, **kwargs):
        scans.append(epochs.size)
        return original(scanner, epochs, *args, **kwargs)

    monkeypatch.setattr(auto_init.PSPLFFTScanner, "search_tE", count)
    monkeypatch.setattr(auto_init, "_last_search", None)
    initial = curve.fit(mask=excluded)
    changed = excluded.copy()
    changed[50] = True
    curve.fit(mask=changed, start=initial.params)
    repeated = curve.fit(mask=excluded)
    np.testing.assert_allclose(repeated.params, initial.params)
    assert scans == [np.count_nonzero(~excluded)]
    curve.fit(mask=changed)
    assert scans == [np.count_nonzero(~excluded), np.count_nonzero(~changed)]
    flux[100] += 0.02
    fitter.prepare(time, flux, ferr).fit(mask=changed)
    assert len(scans) == 3


def test_seeded_fspl_recovers_radius_outside_source_transit():
    from singlelensfitter.host_fitting import FSPLFitter

    fitter = FSPLFitter(maxiter=500, tol=1e-10)
    time = np.linspace(90, 110, 300)
    truth = [100, 4, 0.15, 0.08]
    flux = 1.7 * fitter.magnification_model(time)([*truth[:3], np.log(truth[3])]) + 0.2
    result = fitter.fit(time, flux, np.full_like(time, 0.01),
                        [100.2, 3.5, 0.18, np.log(0.1)])
    np.testing.assert_allclose(result.params, truth, atol=1e-4)


def test_masked_host_fspl_starts_from_shared_fft(monkeypatch):
    from singlelensfitter import fspl_fft
    from singlelensfitter.host_fitting import FSPLFitter

    fitter = FSPLFitter()
    time = np.linspace(-2., 2., 151)
    flux = fitter.magnification_model(time)([.1, 1., .15, np.log(.4)])
    mask = time < -1.5
    seen = []
    search = fspl_fft.fft_seeds

    def record(t, *args):
        seen.append(t.copy())
        return search(t, *args)

    monkeypatch.setattr(fspl_fft, "fft_seeds", record)
    fit = fitter.fit(time, flux, np.full_like(time, .01), mask=mask,
                     pspl_params=[-1., 5., .02])
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], time[~mask])
    assert fit.chi2 < 1e-4
