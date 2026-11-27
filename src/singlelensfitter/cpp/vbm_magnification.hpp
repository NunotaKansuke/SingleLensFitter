#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "VBMicrolensingLibrary.h"
#include "linear_limb_darkening.hpp"
#include "uniform_limb_table.hpp"

namespace singlelensfitter {

// VBM indexes its ESPL table with -log(0.01 * rho).
constexpr double max_espl_rho = 100.0;

inline double espl_radius_from_log(double log_rho) {
    // exp(log(100)) can round above the final table node.
    return log_rho == std::log(max_espl_rho) ? max_espl_rho : std::exp(log_rho);
}

class VbmMagnification {
public:
    VbmMagnification(bool finite_source, const std::string& espl_table, double tol, double reltol, double a = 0)
        : finite_source_(finite_source), a_(a), vbm_(std::make_unique<VBMicrolensing>()) {
        configure(tol, reltol, a);
        if (finite_source_ && !espl_table.empty()) {
            vbm_->LoadESPLTable(espl_table.c_str());
            load_table_copy(espl_table);
        }
    }

    void configure(double tol, double reltol, double a) {
        if (!(a >= 0 && a <= 1) || (!finite_source_ && a != 0))
            throw std::invalid_argument("linear_limb_darkening requires a finite source and 0 <= a <= 1");
        a_ = a;
        vbm_->Tol = tol;
        vbm_->RelTol = reltol;
    }

    double operator()(double u, double rho) {
        if (!std::isfinite(u) || u < 0.0) throw std::runtime_error("invalid lens separation");
        if (finite_source_ && !(rho > 0.0 && rho <= max_espl_rho))
            throw std::invalid_argument("ESPL requires 0 < rho <= 100");
        return evaluate<false>(u, rho).value;
    }

    void value_and_gradient(double u, double rho, double& value, double& d_du, double& d_drho) {
        if (!(std::isfinite(u) && u >= 0 && (!finite_source_ || (rho > 0 && rho <= max_espl_rho))))
            throw std::invalid_argument("invalid separation or source radius");
        const auto m = evaluate<true>(u, rho);
        value = m.value; d_du = m.du; d_drho = m.drho;
    }

    void gradient(double u, double rho, double& d_du, double& d_drho) {
        if (!finite_source_) {
            const double u2 = u*u, h2 = u2+4;
            d_du = -8/(u2*h2*std::sqrt(h2));
            d_drho = 0;
            return;
        }
        const auto m = evaluate<true>(u, rho);
        d_du = m.du; d_drho = m.drho;
    }

    // Epochs away from a transit often form long contiguous runs. A uniform
    // quadrupole satisfies 10*quad < 40*rho^2/u^6, so every member of such
    // a run uses the same VBM branch, independently of limb darkening.
    template<bool derivatives, bool coefficient>
    void batch(const double* input, size_t n, double rho, double* value,
               double* du=nullptr, double* dr=nullptr, double* da=nullptr) {
        const double limit=std::max(64*rho, std::pow(40*rho*rho/vbm_->Tol, 1./6));
        size_t i=0;
        while (i<n) {
            const size_t begin=i;
            while (i<n && rho>=1e-12 && std::abs(input[i])>limit && std::abs(input[i])<1e150) ++i;
#if defined(_OPENMP) && _OPENMP >= 201307
            #pragma omp simd
#endif
            for (size_t j=begin; j<i; ++j) {
                double slope=0;
                const auto m=wing<derivatives,coefficient>(std::abs(input[j]),rho,slope);
                value[j]=m.value;
                if constexpr (derivatives) { du[j]=m.du; dr[j]=m.drho; }
                if constexpr (coefficient) da[j]=slope;
            }
            if (i<n) {
                const double u=std::abs(input[i]);
                if (!std::isfinite(u)) throw std::invalid_argument("u must contain only finite values.");
                const auto m=evaluate<derivatives>(u,rho,coefficient ? da+i : nullptr);
                if (!(m.value>0 && std::isfinite(m.value)))
                    throw std::runtime_error("compiled finite-source magnification is invalid.");
                value[i]=m.value;
                if constexpr (derivatives) { du[i]=m.du; dr[i]=m.drho; }
                ++i;
            }
        }
    }

private:
    template<bool derivatives, bool coefficient>
    SourceMoment wing(double u, double rho, double& da) const {
        const double inv_u=1/u, inv_h=1/std::sqrt(u*u+4);
        const double v=u*inv_h, q=v*v, g=8*inv_h*inv_h*inv_h*inv_u;
        const double size=rho*inv_u, size2=size*size;
        const double first=size2*(1+3*q);
        const double disk=first/8;
        const double gamma=2*a_/(3-a_);
        double corr=disk, radial=2*disk, dq=3*size2/8;
        if (gamma!=0 || coefficient) {
            // For z>=64 the first omitted mu term is O(z^-8).
            const double s4=size2*size2/280, s6=s4*size2/6;
            const double p2=9+q*(15+q*(15+105*q));
            const double p3=25+q*(39+q*(50+q*(70+q*(-315+1155*q))));
            const double second=s4*p2, third=s6*p3;
            const double difference=-first/40+second+third;
            corr+=gamma*difference;
            if constexpr (coefficient) da=6*g*difference/((3-a_)*(3-a_));
            if constexpr (derivatives) {
                radial+=gamma*(-first/20+4*second+6*third);
                dq+=gamma*(-3*size2/40+s4*(15+q*(30+315*q))
                     +s6*(39+q*(100+q*(210+q*(-1260+5775*q)))));
            }
        }
        SourceMoment result;
        result.value=v+2*inv_h*inv_u+g*corr;
        if constexpr (derivatives) {
            result.du=g*inv_u*(-1-(1+3*q)*corr-radial+2*q*(1-q)*dq);
            result.drho=(g*radial)/rho;
        }
        return result;
    }

    template<bool derivatives>
    SourceMoment evaluate(double u, double rho, double* da=nullptr) {
        const auto uniform = uniform_disk<derivatives>(u,rho);
        if (a_ == 0 && !da) return uniform;
        const auto mu = mu_moment<derivatives>(u,std::max(rho,1e-12));
        const double gamma=2*a_/(3-a_);
        if (da) *da=6*(mu.value-uniform.value)/((3-a_)*(3-a_));
        return {(1-gamma)*uniform.value+gamma*mu.value,
                (1-gamma)*uniform.du+gamma*mu.du,
                (1-gamma)*uniform.drho+gamma*mu.drho};
    }

    // Only called outside the limb (u/r > 1.05). For the uniform basis the
    // VBM annulus integrator returns its first ESPL interpolation. Evaluate
    // that same table directly, reusing the radial coordinate for the batch.
    template<bool derivatives>
    SourceMoment exterior_uniform(double u, double r) {
        SourceMoment out;
        const double u2 = u*u, h2 = u2+4, h = std::sqrt(h2);
        const double quad = 4*(u2+1)*(r*r)/(u2*u*h2*h2*h);
        const double pref = (u2+2)/(u*h);
        double point_du = 0;
        if constexpr (derivatives) point_du = -8/(u2*h2*h);
        if (10*quad < vbm_->Tol) {
            out.value = pref + quad;
            if constexpr (derivatives) {
                out.du = point_du + quad*(2*u/(u2+1) - 3/u - 5*u/h2);
                out.drho = 2*quad/r;
            }
            return out;
        }
        if (table_out_.empty()) {
            if constexpr (derivatives) throw std::runtime_error("ESPL table was not loaded");
            out.value = vbm_->ESPLMag2(u, r);
            return out;
        }
        if (r != row_radius_) {
            constexpr double scale = 10.857362047581296;
            const double raw = -scale*std::log(.01*r);
            const double fr = std::clamp(raw, 0., __rsize_ESPL-1.000001);
            row_ = static_cast<size_t>(fr)*__zsize_ESPL;
            radial_fraction_ = fr - static_cast<int>(fr);
            radial_derivative_ = fr == raw ? -scale/r : 0.;
            row_radius_ = r;
        }
        const double z = (.99999999999999/(u/r))*(__zsize_ESPL-1);
        const auto i = static_cast<size_t>(z);
        const double fz = z-i, fr = radial_fraction_;
        const double* lo = table_out_.data()+row_+i;
        const double* hi = lo+__zsize_ESPL;
        const double corr = lo[0]*(1-fr)*(1-fz) + hi[0]*fr*(1-fz) + lo[1]*(1-fr)*fz + hi[1]*fr*fz;
        out.value = ((u2+2)/std::sqrt(u2*h2))*corr;
        if constexpr (derivatives) {
            const double dc_fr = (hi[0]-lo[0])*(1-fz) + (hi[1]-lo[1])*fz;
            const double dc_fz = (lo[1]-lo[0])*(1-fr) + (hi[1]-hi[0])*fr;
            out.du = point_du*corr + pref*dc_fz*(-z/u);
            out.drho = pref*(dc_fr*radial_derivative_ + dc_fz*(z/r));
        }
        return out;
    }

    template<bool derivatives>
    SourceMoment uniform_disk(double u, double rho) {
        const double r = std::max(rho, 1e-12);
        const double z = u/r;
        SourceMoment table, limb;
        if (!finite_source_ || z > 1.05) {
            if (finite_source_) {
                table = exterior_uniform<derivatives>(u, r);
            } else {
                table.value = vbm_->PSPLMag(u);
                if constexpr (derivatives) {
                    const double u2 = u*u, h2 = u2+4;
                    table.du = -8/(u2*h2*std::sqrt(h2));
                }
            }
            if (!(table.value > 0 && std::isfinite(table.value)))
                throw std::runtime_error("VBMicrolensing returned an invalid magnification");
            if (!finite_source_ || z >= 1.15) return table;
        }
        const auto integral = uniform_limb<derivatives>((u-r)/r, r);
        limb.value = integral.value;
        if constexpr (derivatives) {
            limb.du = u == 0 ? 0 : integral.dz/r;
            limb.drho = integral.drho_at_z - z*limb.du;
        }
        if (z <= 1.05) return limb;
        // Smoothstep gives zero weight derivative at both joins.
        const double t = (z-1.05)/0.10;
        const double w = t*t*(3-2*t);
        const double difference = table.value-limb.value;
        if constexpr (derivatives) {
            const double dw_du = 60*t*(1-t)/r;
            limb.du += w*(table.du-limb.du) + dw_du*difference;
            limb.drho += w*(table.drho-limb.drho) - z*dw_du*difference;
        }
        limb.value += w*difference;
        return limb;
    }

    // VBM keeps its ESPL tables private.  The file holds the "in" and "out"
    // magnification blocks first (151 x 101 doubles each); a private copy lets
    // value_and_gradient differentiate the same bilinear interpolant.
    void load_table_copy(const std::string& path) {
        constexpr size_t block = static_cast<size_t>(__rsize_ESPL) * __zsize_ESPL;
        table_in_.resize(block);
        table_out_.resize(block);
        std::FILE* file = std::fopen(path.c_str(), "rb");
        if (!file) throw std::runtime_error("cannot open ESPL table: " + path);
        const bool ok = std::fread(table_in_.data(), sizeof(double), block, file) == block
            && std::fread(table_out_.data(), sizeof(double), block, file) == block;
        std::fclose(file);
        if (!ok) throw std::runtime_error("ESPL table is truncated: " + path);
    }

    double row_radius_ = -1, radial_fraction_ = 0, radial_derivative_ = 0;
    size_t row_ = 0;
    std::vector<double> table_in_, table_out_;
    bool finite_source_ = false;
    double a_;
    std::unique_ptr<VBMicrolensing> vbm_;
};

}  // namespace singlelensfitter
