# SingleLensFitter benchmarks

## Light-curve fits

One synthetic event per row, with 1,000 evenly spaced epochs from −12 to +12
days, constant flux errors of 0.02, and a fixed Gaussian noise realization.
VBMicrolensing (VBM) generated the data. The finite-source event crosses the
source limb. The linear limb-darkening coefficient is fixed at 0.5.

| Model | SLF automatic | SLF seeded | SLF model + FD | VBM + FD | χ² SLF seeded / VBM + FD |
|---|---:|---:|---:|---:|---:|
| PSPL | 30.7 ms | 1.04 ms | 1.22 ms | 2.43 ms | 958.90 / 958.90 |
| Uniform FSPL | 31.5 ms | 1.96 ms | 2.63 ms | 6.48 ms | 961.62 / 959.90 |
| Linear FSPL | 35.8 ms | 2.09 ms | 3.17 ms | 29.66 ms | 960.84 / 959.87 |

The three seeded columns use the same starting geometry, centered peak time,
log timescales and SciPy LM tolerances (10⁻⁶). Source and blend fluxes are
solved by weighted least squares for each geometry. FD uses SciPy's two-point
finite differences. VBM evaluates a whole light curve in a native batch.
The SLF-model FD column uses SLF magnifications in the same FD driver to
separate derivative costs from differences between model implementations.
Both seeded SLF routes reach χ² values within 0.00001 of each other in all
three cases. VBM and SLF use different finite-source approximations at nominal
magnification tolerances of 10⁻⁴. The χ² difference on VBM-generated data is
specific to this injection.

Times are medians of nine measured calls after one warmup per method, with
rotating method order. Automatic timing includes seed search and multistart
with the last PSPL FFT result cleared before each call. Timings exclude
imports, data generation and fitter construction. The SLF API returns a full
fit result, while the FD driver returns the optimized χ².

## Cross-package comparison

For the final comparison at identical starting parameters and LM stopping
rules, see the [matched seeded fits](comparison.md#matched-seeded-fits). It reports runtime and χ² evaluated
with a common reference model. The earlier single-injection experiment is
available as [VBM-generated measurements](synthetic_packages.csv).

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

## Magnification only

Replay of the singlens batch example on the same CPU core. Median time to
evaluate 1,000 epochs after warmup, using float64 and synchronized JAX calls.
The event has tE = 20 days, u0 = 0.005 and ρ = 0.01. SLF uses its default
tolerances, VBM uses the example's 10⁻³ tolerance, and singlens uses its
table and default FFTLog configurations. [Upstream example](https://github.com/ShotaMiyazaki94/singlens/tree/23bcf7a2a1643a225f32c4e1042ebe4f65d14e8e/example/compare-single-vbml)

| Source | SLF | VBM | singlens table | singlens FFTLog |
|---|---:|---:|---:|---:|
| Uniform | 22.9 µs | 81.7 µs | 91.4 µs | 1,653 µs |
| Linear | 45.5 µs | 635 µs | 89.2 µs | 1,878 µs |

Across the upstream reference grid of 840 points per source profile, the
maximum relative magnification errors for SLF are 2.02 × 10⁻⁵ (uniform) and
1.23 × 10⁻⁵ (linear). [Timings for 10–1,000 epochs](magnification_times.csv)

## Reproduce

Measured on Linux with one Intel Xeon Gold 6530 core. For the fits we used
Python 3.11, NumPy 2.4.6, SciPy 1.17.1, JAX 0.10.2, SLF 0.1.4, VBM 5.7,
MulensModel 3.12.0 and pyLIMA 1.9.8. Replace CPU 28 with an available core.
The optional comparison packages are only needed for these benchmarks.

```sh
python -m pip install SingleLensFitter==0.1.4 VBMicrolensing==5.7 numpy==2.4.6 scipy==1.17.1 jax==0.10.2
python -m pip install MulensModel==3.12.0 pyLIMA==1.9.8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c 28 python benchmarks/fits.py
taskset -c 28 python benchmarks/synthetic_packages.py
taskset -c 28 python benchmarks/ogle.py
taskset -c 28 python benchmarks/ogle.py --native
```

[All individual fit timings and χ² values](fit_times.csv)

For the magnification comparison, check out singlens at the revision linked
above. Run the following command once per engine, replacing ENGINE with
`native`, `vbm`, `table` or `fft`. The singlens engines used a separate
Python 3.13 environment with the upstream checkout on PYTHONPATH, JAX 0.12.1
and NumPy 2.5.5. The SLF and VBM engines used Python 3.11.

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 taskset -c 28 python benchmarks/magnification.py ENGINE /path/to/singlens output
```
