"""Single-lens light-curve fitting."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .auto_init import automatic_seeds, best_fit_over_seeds, pspl_initial_guesses
from .fitters import fspl_magnification_gradient, FSPLFitter, FSPLParallaxFitter, PSPLFitter, PSPLParallaxFitter
from .parallax_backend import (
    Ephemeris,
    ParallaxDiagnostics,
    ParallaxEvaluator,
    TimeSpec,
    parallax_effect_score,
)
from .pspl_fft import (
    PSPLFFTCandidate,
    PSPLFFTProfile,
    PSPLFFTScanner,
    PSPLFFTSearchResult,
    pspl_excess_magnification,
)
from .jax_fspl import jax_fspl_magnification
from .init_config import InitConfig
from .singlelens_fit import SingleLensFitResult

__all__ = [
    "jax_fspl_magnification", "fspl_magnification_gradient", "automatic_seeds", "best_fit_over_seeds", "pspl_initial_guesses",
    "FSPLFitter", "FSPLParallaxFitter", "PSPLFitter", "PSPLParallaxFitter",
    "InitConfig", "Ephemeris", "ParallaxDiagnostics", "ParallaxEvaluator", "TimeSpec",
    "parallax_effect_score", "PSPLFFTCandidate", "PSPLFFTProfile", "PSPLFFTScanner",
    "PSPLFFTSearchResult", "pspl_excess_magnification", "SingleLensFitResult",
]
