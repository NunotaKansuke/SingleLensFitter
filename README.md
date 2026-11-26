# SingleLensFitter

Fit single-band gravitational microlensing light curves with a point lens and
either a point source (PSPL) or a finite source (FSPL). Both fitters support
annual and satellite parallax. Finite-source models also support linear limb darkening.
SingleLensFitter began as the single-lens fitting code in `jacscanomaly`.

## Install

Requires Python 3.10+ and a C++17 compiler with OpenMP. On macOS, set
`SINGLELENSFITTER_DISABLE_OPENMP=1` to build with Apple Clang.

```sh
python -m pip install git+https://github.com/NunotaKansuke/SingleLensFitter.git
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

## How the fit works

For a point source, an FFT template search finds starting values for the peak
time, Einstein time, and impact parameter. For a finite source, the fitter
uses the observed peak width to try combinations of source radius (`rho`) and
scaled impact parameter (`u0/rho`), and ranks these starting points by chi-square.

The fitter refines multiple starting points by weighted least squares using
model derivatives. At each step it solves for the source and blend fluxes
analytically.

See the [API guide](https://github.com/NunotaKansuke/SingleLensFitter/blob/main/docs/api.md)
for point-source fits, limb darkening, parallax, and derivatives.
