# How SingleLensFitter fits a light curve

## Finding starting parameters with an FFT

### The score for one trial peak time

Fix the template shape and put its peak at $\tau$. Its excess magnification at
observation $i$ is $T_i=A(t_i-\tau)-1$. With source flux $F_s$, blend flux $F_b$,
and baseline $F_{\rm base}=F_s+F_b$, the predicted flux is $F_{\rm base}+F_sT_i$.
For measured fluxes $F_i$ and weights $w_i=1/\sigma_i^2$, the objective is

$$
\chi^2(F_{\rm base},F_s;\tau)
=\sum_i w_i\left[F_i-F_{\rm base}-F_s T_i\right]^2.
$$

Write $W=\sum_i w_i$, $\bar F=\sum_i w_i F_i/W$, and $\bar T=\sum_i w_i T_i/W$.
Setting the derivative with respect to $F_{\rm base}$ to zero gives
$F_{\rm base}=\bar F-F_s\bar T$. Substituting this baseline and expanding the square gives

$$
\begin{aligned}
\chi^2(F_s;\tau)
&=\sum_i w_i\left[(F_i-\bar F)-F_s(T_i-\bar T)\right]^2 \\
&=\chi^2_{\rm flat}-2F_s C+F_s^2 D,
\end{aligned}
$$

where $\chi^2_{\rm flat}=\sum_i w_i(F_i-\bar F)^2$ is the best constant-flux fit,
$C=\sum_i w_i(F_i-\bar F)T_i$, and $D=\sum_i w_i(T_i-\bar T)^2$.
For a nonsingular shift, minimizing this quadratic subject to $F_s\geq0$
gives $\widehat F_s=\max(0,C)/D$. The reduction in chi-square is therefore

$$
\boxed{\Delta\chi^2(\tau)=\chi^2_{\rm flat}-\chi^2_{\min}(\tau)
=\frac{\max(0,C)^2}{D}}.
$$

### Scanning all peak times

Only $C$ and $D$ change as the template slides. On the regular search grid,
let $W_j=\sum_{i\in j}w_i$ and $X_j=\sum_{i\in j}w_i(F_i-\bar F)$; empty bins
contribute zero to both. At trial grid index $k$, the template value in bin $j$
is $T_{j-k}$. Expanding the centered square in $D$ leaves three correlations:

$$
C_k=\sum_j X_jT_{j-k},\qquad
U_k=\sum_j W_jT_{j-k},\qquad
V_k=\sum_j W_jT_{j-k}^2,\qquad
D_k=V_k-U_k^2/W.
$$

Each correlation gives a sum for every shift in one transform calculation.
For example, $C=\mathcal F^{-1}[\mathcal F(X)\,\overline{\mathcal F(T)}]$,
where the overline denotes complex conjugation and the template covers the
required positive and negative lags. Substituting $C_k,D_k$ into the boxed
expression gives the score at every valid grid peak time. Since
$\chi^2_{\rm flat}$ is fixed, the largest score identifies the smallest fitted
$\chi^2$ for this gridded template.

Across templates, the fitter keeps high-scoring candidates supported by nearby
observations. The shift supplies $t_0$ and the template supplies the starting
geometry: $t_E,u_0$ for a point source, or $\rho,u_0/\rho,t_\star=\rho t_E$
for a finite source. The finite-source search also tests point-source templates.
[FFT scoring](../src/singlelensfitter/pspl_fft.py) · [Finite-source templates](../src/singlelensfitter/fspl_fft.py)

## Refining the fit at the observed times

For each starting point, the fitter evaluates the model at the original observation
times. At each trial lens geometry it solves for the source and blend fluxes by
weighted linear least squares, using $F_i=F_s A(t_i)+F_b$. This leaves only the
nonlinear parameters for the optimizer.

The standard rectilinear fit uses Levenberg–Marquardt with a Jacobian that accounts
for the fitted fluxes. Fitting the limb-darkening coefficient uses a bounded
trust-region solver. If the initial finite-source fits fail to converge or leave
a large $\chi^2$, the fitter also tries starts based on the observed event width.
Supplying an explicit start skips the FFT search. [Fitter implementation](../src/singlelensfitter/fitters.py)

## Finite sources and linear limb darkening

For the linear brightness law $I(\mu)\propto 1-a(1-\mu)$, the normalized disk
is a weighted sum of two profiles. Let $A_{\rm uniform}$ be the magnification
of a uniform disk and $A_\mu$ that of a disk with brightness proportional to
$\mu$, the cosine of the angle between the surface normal and the line of sight.
With each profile normalized to unit flux, the magnification is

$$
A=(1-\Gamma)A_{\rm uniform}+\Gamma A_\mu,\qquad \Gamma=\frac{2a}{3-a}.
$$

This decomposition is exact for a linear brightness law. It also makes the
coefficient derivative inexpensive, since it depends only on the difference
between the two magnifications. A microlensing application of the linear
decomposition appears in [Jiang et al. (2004), Eq. 9](https://www.astronomy.ohio-state.edu/gaudi.1/ob03238.pdf).

The numerical kernel uses piecewise polynomial tables to evaluate values and
derivatives with respect to separation $u$ and radius $\rho$. Near limb contact
($u=\rho$), where the slope changes sharply, it uses dedicated limb calculations
and direct integration where needed. In transition regions it blends the values
and differentiates the blend. [Magnification kernel](../src/singlelensfitter/cpp/vbm_magnification.hpp) · [Limb-darkened basis](../src/singlelensfitter/cpp/linear_limb_darkening.hpp)
