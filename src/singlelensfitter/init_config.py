"""Settings of the automatic single-lens initialization.

Field names and defaults are those of the ``auto_init_*`` block of
``jacscanomaly.FinderConfig``, so a ``FinderConfig`` can still be passed to
the initialization functions (they only read these attributes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InitConfig:
    auto_init_teff_min: float = 0.03
    """Smallest teff used by legacy initialization and the PSPL flat fallback."""

    auto_init_teff_max: float = 100.0
    """Largest teff used by legacy initialization and the PSPL flat fallback."""

    auto_init_teff_grid_n: int = 24
    """Number of logarithmic teff templates used by non-PSPL initialization."""

    auto_init_dt0_coeff: float = 0.25
    """Legacy t0 grid spacing coefficient used for non-PSPL initialization."""

    auto_init_max_clusters: int = 1
    """Maximum number of scan clusters used as t0/teff seeds."""

    auto_init_min_n_eff: float = 2.0
    """
    Minimum effective number of contributing points required by the legacy
    non-PSPL initial grid search.

    This suppresses initial guesses driven by one unrealistically high-weight
    data point.
    """

    auto_init_u0_min: float = 1e-4
    """Smallest u0 template used by the PSPL FFT initial-value search."""

    auto_init_u0_max: float = 1.0
    """Largest u0 template used by the PSPL FFT initial-value search."""

    auto_init_u0_grid_n: int = 8
    """Number of source-plane u0 rows evaluated together for each PSPL tE."""

    auto_init_fft_grid_dt: Optional[float] = 0.02
    """Regular FFT time spacing used for PSPL initialization."""

    auto_init_fft_max_grid_points: int = 500_000
    """Maximum regular FFT grid length used for PSPL initialization."""

    auto_init_fft_top_k: int = 4
    """Number of ranked PSPL FFT seeds passed to the fitter."""

    auto_init_max_flux_cancellation_ratio: float = 50.0
    """
    Maximum source/blend cancellation ratio allowed for an automatic PSPL seed.

    The FFT initializer profiles ``fs`` and ``fb`` independently.  A trial
    whose two terms are much larger than the observed baseline can therefore
    obtain a large apparent improvement while placing ``t0`` in an unobserved
    gap.  Such a seed is not useful to the continuous fitter.
    """

    auto_init_nearest_support_tE_coeff: float = 1.0
    """
    Require one observation within this many ``tE`` of an automatic ``t0``.

    This is a local-support guard in addition to the broader
    ``auto_init_t0_support_tE_coeff`` window.  It keeps a one-sided event near a
    season boundary usable while rejecting a model supported only by a distant
    wing across a large gap.
    """

    auto_init_fft_workers: int = -1
    """SciPy worker count for batched PSPL FFTs; -1 uses all available CPUs."""

    auto_init_fft_tE_grid_n: int = 24
    """Number of logarithmic outer tE scales used by the PSPL FFT search."""

    auto_init_tE_min: float = 1.0
    """Smallest tE scale used by PSPL FFT and legacy non-PSPL initialization."""

    auto_init_tE_max: float = 1000.0
    """Largest tE scale used by PSPL FFT and legacy non-PSPL initialization."""

    auto_init_tE_grid_n: int = 4
    """Number of legacy tE seeds used by non-PSPL initialization."""

    auto_init_logrho: float = -7.0
    """Initial logrho used for FSPL models when x0 is omitted."""

    auto_init_fspl_template_top_k: int = 4
    """Number of ranked FSPL template seeds passed to the fitter."""

    auto_init_min_t0_support_points: int = 3
    """Minimum number of observations near an automatic PSPL seed."""

    auto_init_t0_support_tE_coeff: float = 3.0
    """Support half-width for automatic PSPL seeds, in units of tE."""
