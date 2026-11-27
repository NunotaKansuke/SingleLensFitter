"""Ground-only PSPL fit to MulensModel's OGLE-2014-BLG-0939 example data."""
import argparse
import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

import MulensModel as mm
import numpy as np
from pyLIMA import event, telescopes
from pyLIMA.models.PSPL_model import PSPLmodel
from singlelensfitter import PSPLFitter

from fits import fd_fit, measure, vbm_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="compare package-level fitting workflows")
    args = parser.parse_args()
    path = Path(mm.DATA_PATH) / "photometry_files/OB140939/ob140939_OGLE.dat"
    time, mag, err_mag = np.loadtxt(path, usecols=(0, 1, 2), unpack=True)
    flux, err = mm.Utils.get_flux_and_err_from_mag(mag, err_mag)
    # Optimize a time offset so that a finite-difference step is well resolved.
    origin = 2456836.
    epochs = time - origin
    seed = np.array([0., 23., .9])
    slf = PSPLFitter()
    mulens = mm.Model({"t_0": 0., "t_E": 23., "u_0": .9})
    telescope = telescopes.Telescope(
        name="OGLE", lightcurve=np.column_stack((time, flux, err)),
        lightcurve_names=["time", "flux", "err_flux"],
        lightcurve_units=["JD", "W/m^2", "W/m^2"])
    np.testing.assert_array_equal(telescope.lightcurve["time"].value, time)
    np.testing.assert_array_equal(telescope.lightcurve["flux"].value, flux)
    np.testing.assert_array_equal(telescope.lightcurve["err_flux"].value, err)
    np.testing.assert_array_equal(epochs + origin, time)
    data = event.Event()
    data.telescopes.append(telescope)
    pylima = PSPLmodel(data)
    vbm = vbm_curve(epochs, False, 0.)

    def mm_mag(q):
        mulens.parameters.t_0, mulens.parameters.t_E, mulens.parameters.u_0 = q
        return mulens.get_magnification(epochs)

    def pylima_mag(q):
        params = pylima.compute_pyLIMA_parameters([q[0] + origin, q[2], q[1]])
        return pylima.model_magnification(telescope, params)

    def slf_fit():
        result = slf.fit(epochs, flux, err, seed)
        if not result.optimizer_success:
            raise RuntimeError(result.optimizer_status)
        return float(result.chi2)

    cases = {"SLF seeded": slf_fit}
    for name, model in [("VBM", vbm), ("MulensModel", mm_mag), ("pyLIMA", pylima_mag)]:
        cases[name + " + FD"] = lambda model=model: fd_fit(model, flux, err, seed)
    if args.native:
        from pyLIMA.fits.TRF_fit import TRFfit
        from scipy.optimize import minimize

        photometry = mm.MulensData(data_list=[epochs, flux, err], phot_fmt="flux")
        assert photometry.good.all()
        np.testing.assert_array_equal(photometry.time, epochs)
        np.testing.assert_array_equal(photometry.flux, flux)
        np.testing.assert_array_equal(photometry.err_flux, err)
        fit_event = mm.Event(datasets=[photometry], model=mulens)
        trf = TRFfit(pylima)

        def mm_objective(q):
            mulens.parameters.t_0, mulens.parameters.t_E, mulens.parameters.u_0 = q
            return fit_event.get_chi2(), fit_event.get_chi2_gradient(["t_0", "t_E", "u_0"])

        def mm_fit():
            result = minimize(mm_objective, seed, jac=True, method="L-BFGS-B",
                              bounds=[(None, None), (0.01, None), (0.001, None)],
                              options={"ftol": 1e-13, "gtol": 1e-7})
            if not result.success:
                raise RuntimeError(result.message)
            return float(result.fun)

        def pylima_fit():
            trf.model_parameters_guess = [seed[0] + origin, seed[2], seed[1]]
            trf.telescopes_fluxes_parameters_guess = []
            trf.fit_results = {}
            with redirect_stdout(StringIO()):
                trf.fit()
            result = trf.fit_results
            if not result or not result["fit_object"].success:
                raise RuntimeError("pyLIMA TRF fit failed")
            return float(result["chi2"])

        cases = {"SLF seeded": slf_fit, "MulensModel Event + L-BFGS-B": mm_fit,
                 "pyLIMA TRFfit": pylima_fit}
    writer = csv.writer(sys.stdout)
    writer.writerow(["model", "method", "repeat", "ms", "chi2"])
    for name, rows in measure(cases).items():
        for i, (ms, chi) in enumerate(rows, 1):
            writer.writerow(["OGLE PSPL", name, i, ms, chi])


if __name__ == "__main__":
    main()
