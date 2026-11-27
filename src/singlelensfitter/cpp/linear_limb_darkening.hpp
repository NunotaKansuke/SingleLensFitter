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

    constexpr double max_radius = mu_table::radii[std::size(mu_table::radii)-1];
    constexpr double radial_blend = 0.05*max_radius;
    // The contact layer narrows in z as the source grows. Widen the
    // direct-integration guard so its blend does not amplify table roundoff.
    const double excess = std::max(0.,rho-2);
    const double contact = 1e-6*(1+excess*excess);
    // For large disks the small wing derivatives are more accurately
    // evaluated by direct_mu_moment's exterior rules.
    double radial_slope = 0;
    const double z_end = 10-6*smooth_ramp(excess/2,radial_slope);
    if (rho>=max_radius || z>=z_end || gap<=contact)
        return direct_mu_moment<derivatives>(u,rho);
    auto table = interpolated_mu<derivatives>(z,rho);
    if (rho<=max_radius-radial_blend && z<=z_end-1 && gap>=2*contact) return table;

    double dr, dz, dg;
    const double wr=smooth_ramp((max_radius-rho)/radial_blend,dr);
    const double wz=smooth_ramp(z_end-z,dz);
    const double wg=smooth_ramp(gap/contact-1,dg);
    const double w=wr*wz*wg;
    const auto exact=direct_mu_moment<derivatives>(u,rho);
    const double difference=table.value-exact.value;
    if constexpr (derivatives) {
        const double dw_dz=wr*(-dz*wg+wz*dg*(z<1 ? -1 : 1)/contact);
        const double dw_drho_at_z = -dr*wz*wg/radial_blend
            - 3*wr*dz*radial_slope*wg
            - wr*wz*dg*(gap/contact)*(2e-6*excess/contact);
        table.du=exact.du+w*(table.du-exact.du)+dw_dz*difference/rho;
        table.drho=exact.drho+w*(table.drho-exact.drho)
                   +(dw_drho_at_z-z*dw_dz/rho)*difference;
    }
    table.value=exact.value+w*difference;
    return table;
}

}  // namespace singlelensfitter
