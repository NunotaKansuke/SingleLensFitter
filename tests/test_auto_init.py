from types import SimpleNamespace

import numpy as np
import pytest

from singlelensfitter import auto_init, InitConfig, FSPLFitter, FSPLParallaxFitter


def test_seed_budget_distributes_signed_pairs(monkeypatch):
    anchors = tuple(np.array([t, 3., .2, 0., 0.]) for t in range(4))
    monkeypatch.setattr(auto_init, "pspl_initial_guesses", lambda *args, **kwargs: anchors)
    seeds = auto_init.automatic_seeds([], [], [], finite_source=False, parallax=True)
    assert len(seeds) == 24
    for i, (pos, neg) in enumerate(zip(seeds[::2], seeds[1::2])):
        anchor = anchors[i % 4]
        assert pos[0] == anchor[0]
        np.testing.assert_array_equal(neg, pos * [1, 1, -1, 1, 1])
        if i < 4:
            np.testing.assert_array_equal(pos, anchor)
    assert len(set(map(tuple, seeds))) == 24


@pytest.mark.parametrize("count", [1, 4, 8])
def test_configured_fspl_anchors_reach_fitters(count):
    config = InitConfig(auto_init_fspl_template_top_k=count)
    fitter = FSPLFitter(init_config=config)
    t = np.linspace(-1., 1., 101)
    flux = fitter._magnification(t, np.array([.05, 1., .1, np.log(.3)]))
    data = (t, flux, np.full_like(t, .01))
    anchors = fitter._automatic_initial_guesses(*data)
    assert 1 <= len(anchors) <= count + 1
    parallax = FSPLParallaxFitter(270., -30., 0., init_config=config)
    seeds = parallax._automatic_fspl_initial_guesses(*data)
    for anchor in anchors:
        pos = np.r_[anchor, 0., 0.]
        assert any(np.array_equal(seed, pos) for seed in seeds)
        pos[2] *= -1
        assert any(np.array_equal(seed, pos) for seed in seeds)


@pytest.mark.parametrize("retry_status,expected", [(True, 1.), (False, 3.)])
def test_retry_competitive_unfinished_fit(retry_status, expected):
    calls = []
    def fit(seed):
        calls.append(seed)
        return SimpleNamespace(chi2=seed, optimizer_success=(seed == 3 or seed == 1 and retry_status))
    best = auto_init.best_fit_over_seeds(fit, [3., 2., 4.], "test",
                                        seed_from_fit=lambda result: result.chi2 - 1)
    assert calls == [3., 2., 4., 1.]
    assert best.optimizer_success and best.chi2 == expected


def test_failed_retry_keeps_best_unfinished_and_status():
    def fit(seed):
        if seed == 0:
            raise RuntimeError("failed continuation")
        return SimpleNamespace(chi2=seed, optimizer_success=False)
    best = auto_init.best_fit_over_seeds(fit, [4, 2], "test", seed_from_fit=lambda result: 0)
    assert best.chi2 == 2 and not best.optimizer_success


def test_alternate_does_not_displace_converged_fit():
    incumbent = SimpleNamespace(chi2=4., optimizer_success=True)
    result = auto_init.best_fit_over_seeds(
        lambda seed: SimpleNamespace(chi2=1., optimizer_success=False),
        [1], "test", incumbent=incumbent)
    assert result is incumbent


def test_converged_winner_preserves_unfinished_competitor_across_gap_probes():
    from dataclasses import replace
    from singlelensfitter.singlelens_fit import SingleLensFitResult

    a = np.ones(3)
    original = SingleLensFitResult(a, a, a, a, ("t0", "tE", "u0"), 4., 1., 1., 0., a, a,
                                   optimizer_success=True)
    unfinished = replace(original, chi2=1., optimizer_success=False)
    first = auto_init.best_fit_over_seeds(lambda _: unfinished, [0], "test", incumbent=original)
    improved = replace(original, chi2=2.)
    second = auto_init.best_fit_over_seeds(lambda _: improved, [0], "test", incumbent=first)
    assert second.optimizer_success and second.chi2 == 2.
    assert second.unfinished_candidate is unfinished
    winner = replace(original, chi2=.8)
    third = auto_init.best_fit_over_seeds(lambda _: winner, [0], "test", incumbent=second)
    assert third is winner and third.unfinished_candidate is None


def test_blind_fspl_search_runs_with_pspl_hint_and_explicit_seed_skips_it(monkeypatch):
    from singlelensfitter import fspl_fft

    fitter = FSPLFitter()
    t = np.linspace(-3., 3., 601)
    q = np.array([.4, .8, .1, np.log(.5)])
    flux = fitter._magnification(t, q)
    err = np.full_like(t, .01)
    calls = []
    search = fspl_fft.fft_seeds

    def record_search(*args):
        seeds = search(*args)
        calls.append(len(seeds))
        return seeds

    monkeypatch.setattr(fspl_fft, "fft_seeds", record_search)
    blind = fitter.fit(t, flux, err, pspl_params=[-2., 20., .01])
    explicit = fitter.fit(t, flux, err, p0=q)
    assert len(calls) == 1 and calls[0] > 0
    assert blind.chi2 < 1e-4 and explicit.chi2 < 1e-4


def test_blind_fspl_includes_exact_point_source_candidate():
    from singlelensfitter.fspl_fft import fft_seeds
    from singlelensfitter.pspl_fft import pspl_excess_magnification

    t = np.arange(-10., 10., .02)
    flux = 1. + pspl_excess_magnification(t, u0=.5, teff=1.25)
    seeds = fft_seeds(t, flux, np.full_like(t, .01), FSPLFitter(), InitConfig())
    assert any(np.allclose(seed[:3], [0., 2.5, .5]) and np.isfinite(seed[3])
               for seed in seeds)
