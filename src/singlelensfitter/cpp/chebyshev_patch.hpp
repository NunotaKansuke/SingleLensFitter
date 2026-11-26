#pragma once

#include <algorithm>
#include <array>
#include <iterator>

namespace singlelensfitter {

// Clenshaw recurrence for a Chebyshev series and its coordinate derivative.
template<bool derivatives, size_t N>
inline double chebyshev(const double (&c)[N], double x, double& slope) {
    double b1=0, b2=0, d1=0, d2=0;
    for (int j=N-1; j>0; --j) {
        const double b = c[j] + 2*x*b1 - b2;
        if constexpr (derivatives) {
            const double d = 2*b1 + 2*x*d1 - d2;
            d2=d1; d1=d;
        }
        b2=b1; b1=b;
    }
    if constexpr (derivatives) slope = b1 + x*d1 - d2;
    return c[0] + x*b1 - b2;
}

// At a fixed radius, contract the radius dimension once per visited patch.
// Each thread owns its cache. A new radius replaces only the visited slices.
struct PatchValue { double value, dt, drho; };

template<bool derivatives, size_t Patches, size_t R, size_t M, size_t N>
inline PatchValue sample_patch(const double (&radii)[R], const double (&coeff)[M][N][N],
                               size_t patch, double rho, double t) {
    static_assert(M == (R-1)*Patches);
    struct Slice {
        double rho = -1;
        double values[N], radial_slopes[N];
    };
    static thread_local std::array<Slice, Patches> slices{};
    auto& slice = slices[patch];
    if (rho != slice.rho) {
        const auto bin = std::upper_bound(radii+1, radii+R, rho)-radii-1;
        const double scale = 2/(radii[bin+1]-radii[bin]);
        const double x = (rho-radii[bin])*scale-1;
        const auto& c = coeff[bin*Patches+patch];
        for (size_t k=0; k<N; ++k) {
            double column[N], dx=0;
            for (size_t j=0; j<N; ++j) column[j]=c[j][k];
            slice.values[k]=chebyshev<derivatives>(column,x,dx);
            if constexpr (derivatives) slice.radial_slopes[k]=dx*scale;
        }
        slice.rho=rho;
    }
    PatchValue out{};
    out.value=chebyshev<derivatives>(slice.values,t,out.dt);
    if constexpr (derivatives) {
        double ignored=0;
        out.drho=chebyshev<false>(slice.radial_slopes,t,ignored);
    }
    return out;
}

// Smooth transition weights, with the derivative in the ramp coordinate.
inline double smooth_ramp(double x, double& slope) {
    if (x >= 1) { slope=0; return 1; }
    slope=6*x*(1-x);
    return x*x*(3-2*x);
}

}  // namespace singlelensfitter
