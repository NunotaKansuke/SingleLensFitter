import numpy as np

from singlelensfitter.signal_scale import measure_observed_magnification_fwhm


def _event(t, t0=50.0, tE=3.0):
    u = np.hypot((t - t0) / tE, 0.3)
    return 100.0 + 100.0 * (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))


def test_half_maximum_crossing_is_not_interpolated_across_a_gap():
    # The rise is in a 60-day gap: data resume on the declining wing.
    t = np.concatenate([np.arange(0.0, 20.0, 0.1), np.arange(49.0, 70.0, 0.1)])
    y = _event(t)
    scale = measure_observed_magnification_fwhm(t, y, center=49.0, search_half_width=60.0)
    assert scale.valid and scale.censored
    assert scale.left_coverage == 0.0
    # width follows the observed wing, not the 30-day gap
    assert scale.half_width < 5.0


def test_fully_sampled_event_is_unchanged():
    t = np.arange(0.0, 100.0, 0.1)
    y = _event(t)
    scale = measure_observed_magnification_fwhm(t, y, center=50.0, search_half_width=60.0)
    assert scale.valid and not scale.censored
    assert scale.left_coverage == 1.0 and scale.right_coverage == 1.0
    assert abs(scale.t_center - 50.0) < 0.2
