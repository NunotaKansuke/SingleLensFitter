"""Package APIs and independent scoring for the RMDC fit comparison."""
from pathlib import Path
import runpy

import MulensModel as mm
import numpy as np
from pyLIMA import event, telescopes
from pyLIMA.fits.TRF_fit import TRFfit
from pyLIMA.models.FSPLarge_model import FSPLargemodel
from pyLIMA.models.PSPL_model import PSPLmodel


quadrature = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                "tests/limb_reference.py"))["magnification"]


def reference(time, q, a, order):
    u = np.hypot((time-q[0])/q[1], q[2])
    if len(q) == 3:
        return (u*u+2)/(u*np.hypot(u, 2))
    return quadrature(u, q[3], a, order=order)


def score_reference(time, flux, err, q, a):
    mag = reference(time, q, a, 96)
    design = np.column_stack([mag, np.ones_like(mag)]) / err[:, None]
    coef = np.linalg.lstsq(design, flux/err, rcond=None)[0]
    residual = flux/err-design@coef
    return residual @ residual
