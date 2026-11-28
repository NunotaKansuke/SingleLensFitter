# SingleLensFitter

Fit point-source (PSPL) and finite-source (FSPL) microlensing light curves
directly from single-band photometry.
SingleLensFitter uses magnification templates and fast Fourier transforms to find initial
parameters, then refines them with Levenberg–Marquardt optimization using analytic derivatives.
Both support annual and satellite parallax. Finite-source models also support linear limb darkening.
SingleLensFitter began as the single-lens fitting code in `jacscanomaly`.

## Install

Requires Python 3.10+ and a C++17 compiler with OpenMP. On macOS, set
`SINGLELENSFITTER_DISABLE_OPENMP=1` to build with Apple Clang.

```sh
python -m pip install SingleLensFitter
```

## Fit a light curve

Pass one-dimensional arrays of observation times in days, fluxes, and positive
one-sigma flux uncertainties.

```python
from singlelensfitter import FSPLFitter

fit = FSPLFitter().fit(time, flux, flux_err)
print(fit.optimizer_success, dict(zip(fit.param_names, fit.params)))
print(fit.chi2, fit.fs, fit.fb)
```

See the [API guide](https://github.com/NunotaKansuke/SingleLensFitter/blob/main/docs/api.md)
for point-source fits, limb darkening, parallax, and derivatives.
[How the fitter works](docs/algorithm.md) describes the FFT search and optimization.

## Benchmarks

On three simulated events from [RMDC26](https://rges-pit.org/data-challenge/data/) with about 7,700 observations each, seeded fits took
2.49–4.96 ms on one CPU core.

[RMDC26 benchmark results and light curves](benchmarks/comparison.md#rmdc26-seeded-fits)
· [OGLE benchmark](benchmarks/README.md)
