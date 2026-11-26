#pragma once

#include <array>
#include <cmath>

namespace singlelensfitter {

// Green's theorem on the source boundary. In units of the source radius,
// z=u/rho, s=1+z^2-2z*cos(theta), and the angular mean of
// (1-z*cos(theta))*sqrt(1+4/(rho^2*s)) is the disk-average magnification.
// Tanh-sinh nodes resolve the narrow layer near theta=0 as z approaches 1.
struct LimbNode {
    double one_minus_cos, weight;
};

template<int HalfCount>
inline const std::array<LimbNode, 2*HalfCount+1>& limb_nodes() {
    static const auto nodes = [] {
        std::array<LimbNode, 2*HalfCount+1> result;
        constexpr double step = 4.0/HalfCount;
        const double pi = std::acos(-1.0);
        for (int i = 0; i <= 2*HalfCount; ++i) {
            const double t = step * (i - HalfCount);
            const double x = pi * std::sinh(t);
            const double q = std::exp(-std::abs(x));
            const double fraction = (x < 0 ? q : 1.0) / (1.0 + q);
            const double sine = std::sin(0.5 * pi * fraction);
            result[i] = {2*sine*sine, step*pi*std::cosh(t)*q/((1+q)*(1+q))};
        }
        return result;
    }();
    return nodes;
}

struct LimbIntegral {
    double value = 0, dz = 0, drho_at_z = 0;
};

template<bool derivatives, int HalfCount>
inline LimbIntegral limb_sum(double delta, double rho) {
    LimbIntegral result;
    const double z = 1 + delta;
    for (const auto& node : limb_nodes<HalfCount>()) {
        const double s = delta*delta + 2*z*node.one_minus_cos;
        const double b = -delta + z*node.one_minus_cos;
        const double inv = 4 / (rho*rho*s);
        const double h = std::sqrt(1 + inv);
        result.value += node.weight*b*h;
        if constexpr (derivatives) {
            const double k = inv/h;
            result.drho_at_z -= node.weight*b*k/rho;
            result.dz += node.weight*((node.one_minus_cos-1)*h
                         - b*k*(delta+node.one_minus_cos)/s);
        }
    }
    return result;
}

template<bool derivatives>
inline LimbIntegral integrate_limb(double delta, double rho) {
    // At very small normal separation the derivative resolves a much
    // narrower angular layer than the magnification.
    auto result = std::abs(delta) < 1e-5 ? limb_sum<derivatives, 240>(delta, rho)
                                        : limb_sum<derivatives, 80>(delta, rho);
    if constexpr (derivatives) {
        if (delta == 0) {
            // A sharp uniform disk has a logarithmically divergent normal
            // derivative at contact. Supply a symmetric secant to the fitter.
            constexpr double step = 1e-6;
            result.dz = (limb_sum<false, 240>(step, rho).value
                       - limb_sum<false, 240>(-step, rho).value)/(2*step);
        }
    }
    return result;
}

}  // namespace singlelensfitter
