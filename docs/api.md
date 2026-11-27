# Fitting API

Import the fitters from `singlelensfitter`. Each fits a single photometric
series and solves for source and blend flux at each set of lens parameters,
using the supplied one-sigma errors as weights in chi-square.

| Fitter | Lens parameters returned in order |
| --- | --- |
| `PSPLFitter()` | `t0, tE, u0` |
| `FSPLFitter()` | `t0, tE, u0, rho`, then the limb-darkening coefficient if fitted |
| `PSPLParallaxFitter(ra_deg, dec_deg, tref)` | `t0, tE, u0, piEN, piEE` |
| `FSPLParallaxFitter(ra_deg, dec_deg, tref)` | `t0, tE, u0, rho, piEN, piEE` |

### Inputs and results

Call `fitter.fit(time, flux, ferr)` with finite one-dimensional arrays of equal
length. Time, peak time `t0`, and Einstein time `tE` are in days. The impact
parameter `u0` and source radius `rho` are in units of the angular Einstein radius.
Flux and its positive uncertainty must share units.

Omit the initial parameters to run automatic multistart. To continue an
existing fit, use `fitter.fit(time, flux, ferr, fitter.seed_from_fit(result))`.
Finite-source seeds use `log(rho)`; results report physical `rho`. Pass an
`InitConfig` as `init_config` to customize automatic initialization.

`SingleLensFitResult` includes `param_names`, `params`, source flux `fs`,
blend flux `fb`, `model_flux`, `residual`, and `chi2`. Check
`optimizer_success` and `optimizer_status` before interpreting parameters.
`chi2_dof` divides by the number of observations minus the number of nonlinear
parameters (minimum denominator one). For nominal residual degrees of freedom,
also count the two fitted flux coefficients in a full-rank fit.

### Limb darkening and derivatives

Use `FSPLFitter(linear_limb_darkening=0.3)` to fix the coefficient in
`I(mu)/I(1) = 1 - a + a*mu`. Zero gives a uniform disk. Add
`fit_limb_darkening=True` to fit the coefficient over [0, 1] for a rectilinear
trajectory. The parallax fitter accepts a fixed coefficient.

```python
from singlelensfitter import fspl_magnification_gradient, jax_fspl_magnification

A, dA_du, dA_drho, dA_da = fspl_magnification_gradient(u, rho, 0.3)
A_jax = jax_fspl_magnification(u, rho, 0.3)
```

Supply nonnegative separations and a scalar source radius up to 100. At exact
limb contact the uniform-disk normal derivative uses a finite symmetric secant.
The JAX function supports JIT and first derivatives. Importing this package
enables JAX 64-bit mode.

### Parallax

Supply sky coordinates in degrees and a reference epoch in the same time
convention as the observations. The default uses full JD and the bundled
Earth ephemeris. For times expressed as JD minus 2450000, for example, use:

```python
from singlelensfitter import PSPLParallaxFitter, TimeSpec

fitter = PSPLParallaxFitter(ra_deg, dec_deg, tref,
                            time_spec=TimeSpec(scale="jd", offset=2450000))
fit = fitter.fit(time, flux, ferr)
```

For a satellite's geocentric offsets, pass `geometry="space"` and
`satellite_or_observer_ephemeris=Ephemeris(times, xyz, origin="earth",
time_spec=spec)`. Use ICRF/J2000 Cartesian positions shaped (N, 3) in AU
and set `spec` to the ephemeris time convention. The ephemeris must cover
the requested epochs. Parallax results include optimizer and ephemeris
information in `fit.diagnostics`.

### Repeated fits with different masks

Prepare the observations once with any of the four fitters. A boolean mask has
one entry per observation; `True` excludes that point from the optimization,
the source/blend flux solution, and chi-square. Predictions and residuals in the
result cover all original epochs.

```python
curve = fitter.prepare(time, flux, ferr)
first = curve.fit()
refit = curve.fit(mask=excluded, start=first)
chi2, grad = curve.chi2_and_grad(refit.params, mask=another_mask)
```

Omitting `start` runs automatic initialization on the retained points.
A result or a vector in physical `result.params` order starts a local fit.
The gradient uses the same physical coordinates (including physical `tE`,
`rho`, and the fitted limb-darkening coefficient). Scoring a new mask at
the same parameters reuses the magnification calculation. Parallax fits
also reuse the projection at every epoch. A prepared curve owns its input
arrays and mutable evaluation cache; use one instance per concurrent worker.

### Host fitting for jacscanomaly

`singlelensfitter.host_fitting` provides the PSPL, FSPL and parallax fitters
used by jacscanomaly. They share the standalone numerical kernels and retain
the host search's seed order and lowest-χ² selection. Prepare each light curve
once. Use `curve.fit(mask=excluded)` for a fresh search on the kept samples
and `curve.fit(mask=excluded, start=fit.params)` to continue a previous fit.
The latter skips template search. Returned models cover all epochs.

For CPU workflows, pass `numpy_result=True` to `PSPLFitter` or `FSPLFitter`
when calling `fit` or `evaluate_fixed` to return fitted values as NumPy arrays.
The parallax fitters already return host-side values.
