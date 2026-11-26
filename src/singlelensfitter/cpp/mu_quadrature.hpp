#pragma once

#include "source_limb.hpp"
#include <algorithm>

namespace singlelensfitter {

struct SourceMoment {
    double value = 0;
    double du = 0;
    double drho = 0;
};

// Normalize the mu = sqrt(1-r^2) brightness basis to unit unlensed flux.
// The near-field rays start at the lens. Differentiate the translated,
// dilated brightness profile at fixed sky position; it vanishes on the
// boundary. Along a chord r = center+half*cos(psi), |dr|/mu = dpsi cancels
// the integrable 1/mu in the two derivatives.
struct AngularNode { double x, weight, sine, cosine; };

template<int N>
inline const std::array<AngularNode, N>& angular_nodes() {
    static const auto nodes = [] {
        std::array<AngularNode, N> out;
        const double pi = std::acos(-1.);
        for (int i = 0; i < N; ++i) {
            // Gauss-Legendre rule on [0,1], initialized once per order.
            double root = std::cos(pi * (i + 0.75) / (N + 0.5));
            double slope = 0;
            for (int iteration = 0; iteration < 12; ++iteration) {
                double polynomial = 1, previous = 0;
                for (int k = 1; k <= N; ++k) {
                    const double next = ((2*k - 1)*root*polynomial - (k - 1)*previous) / k;
                    previous = polynomial;
                    polynomial = next;
                }
                slope = N * (previous - root*polynomial) / (1 - root*root);
                const double step = polynomial / slope;
                root -= step;
                if (std::abs(step) < 2e-16) break;
            }
            const double x = (1 + root) / 2;
            out[i] = {x, 1 / ((1 - root*root)*slope*slope), std::sin(pi*x), std::cos(pi*x)};
        }
        return out;
    }();
    return nodes;
}

template<bool derivatives, int Order>
inline SourceMoment near_mu_moment(double u, double rho) {
    const double pi = std::acos(-1.), z = u/rho;
    SourceMoment m;
    const bool inside = u < rho;
    const double gap = (rho-u)/rho*(1+z);
    for (int side = 0; side < (inside ? 2 : 1); ++side) {
        for (const auto& n : angular_nodes<Order>()) {
            // Cluster angles near tangency on either side of pi/2.
            const double angle = pi*(side+(1-n.cosine)/2)/2;
            double cos_theta = std::cos(angle);
            double center, half, psi_max;
            double w = 0.75*pi*n.weight*n.sine/rho;
            if (inside) {
                center = z*cos_theta;
                half = std::sqrt(gap+center*center);
                psi_max = std::atan2(std::sqrt(gap),-center);
            } else {
                // z*sin(theta)=sin(angle), with dtheta/dangle = half/center.
                half = cos_theta;
                center = std::sqrt(-gap+half*half);
                cos_theta = center/z;
                w *= half/center;
                psi_max = pi;
            }
            const auto& chord = angular_nodes<Order>();
            double value = 0, du = 0, drho = 0;
            #pragma omp simd reduction(+:value,du,drho)
            for (int i = 0; i < Order; ++i) {
                const auto& p = chord[i];
                const double psi = psi_max*p.x;
                const double r = center+half*(inside ? std::cos(psi) : p.cosine);
                const double mu = half*(inside ? std::sin(psi) : p.sine);
                // This near-field rule has u < 4*rho and rho <= 100.
                const double s = rho*r, h = std::sqrt(s*s+4);
                const double weighted = w*p.weight*psi_max*((s*s+2)/h);
                value += weighted*mu*mu;
                if constexpr (derivatives) {
                    du += weighted*(r*cos_theta-z);
                    drho += weighted*(1-3*mu*mu);
                }
            }
            m.value += value;
            if constexpr (derivatives) {
                m.du += du/rho;
                m.drho += drho/rho;
            }
        }
    }
    if (u == 0) m.du = 0;
    return m;
}

// The azimuthal average of a translated radial kernel has coefficients
// rho^(2k) * Laplacian^k A / (4^k k! (5/2)_k) for the unit-flux mu disk.
// Write each term as g*(rho/u)^(2k)*P_k(q), q=u^2/(4+u^2),
// g=-u*dA_point/du. Applying the Laplacian again acts by
// [2q(1-q) d/dq - (1+3q+2k)] twice on P_k.
template<bool derivatives>
inline SourceMoment exterior_mu(double u, double rho) {
    constexpr double numerator[5][10] = {
        {1,3},
        {9,15,15,105},
        {25,39,50,70,-315,1155},
        {245,375,477,567,567,6237,-33033,45045},
        {1323,2009,2540,2996,3402,4158,-36036,420420,-1174173,969969}
    };
    constexpr double denominator[5]={10,280,1680,29568,256256};
    const double inv=1/std::hypot(u,2.), ratio=u*inv, q=ratio*ratio;
    const double g=8*inv*inv*inv/u;
    const double size=rho/u, size2=size*size;
    double power=size2, corr=0, radial=0, dq=0;
    for (int k=1; k<=5; ++k) {
        const auto& c=numerator[k-1];
        double poly=c[2*k-1], slope=0;
        for (int j=2*k-2; j>=0; --j) {
            if constexpr (derivatives) slope=slope*q+poly;
            poly=poly*q+c[j];
        }
        const double scale=power/denominator[k-1];
        corr+=scale*poly;
        if constexpr (derivatives) {
            radial+=2*k*scale*poly;
            dq+=scale*slope;
        }
        power*=size2;
    }
    SourceMoment m;
    m.value=ratio+2*inv/u+g*corr;
    if constexpr (derivatives) {
        m.du=(g/u)*(-1-(1+3*q)*corr-radial+2*q*(1-q)*dq);
        m.drho=(g/rho)*radial;
    }
    return m;
}

template<bool derivatives>
inline SourceMoment direct_mu_moment(double u, double rho) {
    const double z = u/rho;
    SourceMoment m;
    if (z >= 9) return exterior_mu<derivatives>(u,rho);
    if (z >= 4) {
        // Smooth, off-disk integrand in source-centered polar coordinates.
        struct DiskNode { double x, y, radius2, weight; };
        static const auto nodes = [] {
            std::array<DiskNode, 8*12> disk{};
            size_t i = 0;
            for (const auto& n : angular_nodes<8>()) {
                const double radius = std::sqrt(1-n.x*n.x);
                for (const auto& p : angular_nodes<12>()) {
                    disk[i++] = {radius*p.cosine, radius*p.sine, radius*radius,
                                 3*n.weight*n.x*n.x*p.weight};
                }
            }
            return disk;
        }();
        for (const auto& node : nodes) {
            const double dx = u-rho*node.x, dy = rho*node.y;
            // Here 4*rho <= u < 9*rho and 1e-12 <= rho <= 100.
            const double d = std::sqrt(dx*dx+dy*dy), h = std::sqrt(d*d+4);
            m.value += node.weight*(d/h+2/(d*h));
            if constexpr (derivatives) {
                const double slope = -8/(d*d*h*h*h);
                m.du += node.weight*slope*(dx/d);
                m.drho += node.weight*slope*((rho*node.radius2-u*node.x)/d);
            }
        }
        return m;
    }
    if (std::abs(z-1) < 2e-5)
        return near_mu_moment<derivatives, 96>(u,rho);
    if (rho > 10.)
        return near_mu_moment<derivatives, 64>(u, rho);
    return near_mu_moment<derivatives, 32>(u, rho);
}

}  // namespace singlelensfitter
