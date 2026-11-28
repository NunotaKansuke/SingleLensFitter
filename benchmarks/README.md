# SingleLensFitter benchmarks

[RMDC26](https://rges-pit.org/data-challenge/data/) · [Seeded fits and light curves](comparison.md#rmdc26-seeded-fits)

## OGLE-2014-BLG-0939

A seeded, ground-only PSPL fit using all 485 OGLE epochs bundled with
MulensModel gives **χ² = 1269.50241** for every method. This is a test of
fitting the same photometry. The published event analysis also uses Spitzer
and parallax. [Yee et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...76Y/abstract)

| Method | Median fit time |
|---|---:|
| SLF seeded | 0.76 ms |
| VBM magnification + FD | 1.13 ms |
| MulensModel magnification + FD | 1.71 ms |
| pyLIMA magnification + FD | 1.51 ms |

The three FD entries use each package's PSPL magnification function and the
same optimizer, flux profiling, seed and errors. They measure this particular
driver. Package-specific fitters, priors, and automatic initialization are
outside this comparison. The script reads the original photometry from the
installed MulensModel package.

On the same 485 epochs, fluxes and uncertainties, a second run used each
package at the fit level. All started at JD 2456836, tE = 23 days and u0 = 0.9,
used every epoch, and reached χ² = 1269.50241. Median times over nine repeats
were 0.86 ms for SLF, 4.37 ms for pyLIMA’s built-in TRFfit, and 9.26 ms for
MulensModel’s Event objective and gradient with SciPy L-BFGS-B. These timings
exclude event and fitter construction. pyLIMA fits the flux parameters jointly
and computes a covariance matrix. Solver settings and returned diagnostics
differ across these workflows. [Individual measurements](ogle_workflows.csv)

Run the OGLE comparison with `python benchmarks/ogle.py` or `python benchmarks/ogle.py --native`.
