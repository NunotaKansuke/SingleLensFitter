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
