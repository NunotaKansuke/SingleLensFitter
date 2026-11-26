#pragma once

#include "chebyshev_patch.hpp"
#include "mu_quadrature.hpp"
#include "mu_table_data.hpp"

namespace singlelensfitter {

template<bool derivatives>
inline SourceMoment interpolated_mu(double z, double rho) {
    // Inside, z^2 enforces the zero central slope. Outside, sqrt(z-1)
    // resolves the square-root cusp in the first derivative at contact.
    const bool inside = z < 1;
    const double y = inside ? z*z : std::sqrt(z-1);
    const int cell = std::min(static_cast<int>(2*y), inside ? 1 : 5);
    const int patch = inside ? cell : cell+2;
    const double t = 4*y-2*cell-1;
    const auto fit=sample_patch<derivatives,8>(mu_table::radii,mu_table::coefficients,patch,rho,t);
    SourceMoment m;
    m.value = fit.value/rho;
    if constexpr (derivatives) {
        const double dt_dz = inside ? 8*z : 2/y;
        m.du = fit.dt*dt_dz/(rho*rho);
        m.drho = (fit.drho-m.value)/rho-z*m.du;
    }
    return m;
}

template<bool derivatives>
inline SourceMoment mu_moment(double u, double rho) {
    const double z=u/rho, gap=std::abs(z-1);

    if (rho>=2 || z>=10 || gap<=1e-6)
        return direct_mu_moment<derivatives>(u,rho);
    auto table = interpolated_mu<derivatives>(z,rho);
    if (rho<=1.9 && z<=9 && gap>=2e-6) return table;

    double dr, dz, dg;
    const double wr=smooth_ramp((2-rho)*10,dr);
    const double wz=smooth_ramp(10-z,dz);
    const double wg=smooth_ramp((gap-1e-6)*1e6,dg);
    const double w=wr*wz*wg;
    const auto exact=direct_mu_moment<derivatives>(u,rho);
    const double difference=table.value-exact.value;
    if constexpr (derivatives) {
        const double dw_dz=wr*(-dz*wg+wz*dg*(z<1 ? -1e6 : 1e6));
        table.du=exact.du+w*(table.du-exact.du)+dw_dz*difference/rho;
        table.drho=exact.drho+w*(table.drho-exact.drho)
                   +(-10*dr*wz*wg-z*dw_dz/rho)*difference;
    }
    table.value=exact.value+w*difference;
    return table;
}

}  // namespace singlelensfitter
