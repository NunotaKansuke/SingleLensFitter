#pragma once
#include "chebyshev_patch.hpp"
#include "mu_quadrature.hpp"
#include "uniform_table_data.hpp"

namespace singlelensfitter {

template<bool derivatives>
inline LimbIntegral uniform_limb(double delta, double rho) {
    // Interpolate rho*A in log2(abs(z-1)), separately on either side.
    // Retain direct integration at exact contact and outside the table.
    constexpr double inner=0x1p-16, outer=0x1p-4;
    const double gap=std::abs(delta);
    if (gap >= outer && rho <= 1.9 && delta >= -1 && delta <= .15) {
        const double coordinate = 2*(std::log2(gap)+4);
        const int cell = std::min(static_cast<int>(coordinate), delta < 0 ? 7 : 2);
        const int patch = (delta < 0 ? 6 : 14) + cell;
        const auto fit = sample_patch<derivatives,17>(uniform_table::radii,
            uniform_table::coefficients, patch, rho, 2*(coordinate-cell)-1);
        LimbIntegral result;
        result.value = fit.value/rho;
        if constexpr (derivatives) {
            result.dz = fit.dt*4/(std::log(2.)*delta*rho);
            result.drho_at_z = (fit.drho-result.value)/rho;
        }
        return result;
    }
    if constexpr (!derivatives) {
        if (gap >= outer && delta >= -1 && delta <= .15) {
            // The boundary integrand is smooth this far from contact. Set
            // theta = pi*x^2 to resolve its narrowest part near theta = 0.
            static const auto nodes = [] {
                std::array<LimbNode,64> result{};
                const auto& rule = angular_nodes<64>();
                for (size_t i = 0; i < rule.size(); ++i) {
                    const double x = rule[i].x;
                    const double sine = std::sin(.5*std::acos(-1.)*x*x);
                    result[i] = {2*sine*sine, 2*x*rule[i].weight};
                }
                return result;
            }();
            const double z = 1+delta, scale = 4/(rho*rho);
            double value = 0;
#if defined(_OPENMP) && _OPENMP >= 201307
            #pragma omp simd reduction(+:value)
#endif
            for (size_t i = 0; i < nodes.size(); ++i) {
                const double angle = z*nodes[i].one_minus_cos;
                value += nodes[i].weight*(angle-delta)*
                         std::sqrt(1+scale/(delta*delta+2*angle));
            }
            return {value, 0, 0};
        }
    }
    if (gap<=inner || gap>=outer || rho>=2)
        return integrate_limb<derivatives>(delta,rho);
    const double log_gap=std::log2(gap);
    const int octave=std::min(2,static_cast<int>((log_gap+16)/4));
    const int patch=octave+(delta>0 ? 3 : 0);
    const double t=(log_gap+16-4*octave)/2-1;
    const auto fit=sample_patch<derivatives,17>(uniform_table::radii,uniform_table::coefficients,
                                               patch,rho,t);
    LimbIntegral table;
    table.value=fit.value/rho;
    if constexpr (derivatives) {
        table.dz=fit.dt/(2*std::log(2.)*delta*rho);
        table.drho_at_z=(fit.drho-table.value)/rho;
    }
    if (gap>=2*inner && gap<=0.05 && rho<=1.9) return table;
    double di, dout, dr;
    const double wi=smooth_ramp((gap-inner)/inner,di);
    const double wo=smooth_ramp((outer-gap)/(outer-0.05),dout);
    const double wr=smooth_ramp((2-rho)*10,dr);
    const double w=wi*wo*wr;
    const auto direct=integrate_limb<derivatives>(delta,rho);
    const double diff=table.value-direct.value;
    if constexpr (derivatives) {
        const double dw_dz=std::copysign(1.,delta)*wr*(di/inner*wo-wi*dout/(outer-0.05));
        table.dz=direct.dz+w*(table.dz-direct.dz)+diff*dw_dz;
        table.drho_at_z=direct.drho_at_z+w*(table.drho_at_z-direct.drho_at_z)-10*dr*wi*wo*diff;
    }
    table.value=direct.value+w*diff;
    return table;
}

}  // namespace singlelensfitter
