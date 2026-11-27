#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "cpp/ephemeris.hpp"
#include "cpp/parallax_trajectory.hpp"
#include "cpp/vbm_magnification.hpp"

using singlelensfitter::Ephemeris;
using singlelensfitter::ObserverConvention;
using singlelensfitter::ParallaxTrajectory;
using singlelensfitter::TrajectoryComponents;
using singlelensfitter::Vec3;
using singlelensfitter::VbmMagnification;

namespace {

PyArrayObject* as_double_array(PyObject* obj) {
    return reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
}

PyArrayObject* as_int_array(PyObject* obj) {
    return reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
}

bool is_none(PyObject* obj) { return obj == nullptr || obj == Py_None; }

bool read_ephemeris(PyObject* obj, const char* name, Ephemeris& out) {
    if (is_none(obj)) return false;
    PyObject* seq = PySequence_Fast(obj, "ephemeris must be a (time, position, velocity) tuple");
    if (!seq) return false;
    const Py_ssize_t nitems = PySequence_Fast_GET_SIZE(seq);
    if (nitems < 2 || nitems > 3) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "%s must contain time, position and optional velocity", name);
        return false;
    }
    PyObject** items = PySequence_Fast_ITEMS(seq);
    PyArrayObject* t = as_double_array(items[0]);
    PyArrayObject* p = as_double_array(items[1]);
    PyArrayObject* v = nitems == 3 && !is_none(items[2]) ? as_double_array(items[2]) : nullptr;
    if (!t || !p || (nitems == 3 && !is_none(items[2]) && !v)) {
        PyErr_Format(PyExc_ValueError, "invalid %s arrays", name);
        Py_XDECREF(t); Py_XDECREF(p); Py_XDECREF(v); Py_DECREF(seq);
        return false;
    }
    if (PyArray_NDIM(t) != 1 || PyArray_NDIM(p) != 2 || PyArray_DIM(p, 1) != 3 ||
        PyArray_DIM(t, 0) != PyArray_DIM(p, 0) ||
        (v && (PyArray_NDIM(v) != 2 || PyArray_DIM(v, 0) != PyArray_DIM(t, 0) || PyArray_DIM(v, 1) != 3))) {
        PyErr_Format(PyExc_ValueError, "%s arrays have incompatible shapes", name);
        Py_DECREF(t); Py_DECREF(p); Py_XDECREF(v); Py_DECREF(seq);
        return false;
    }
    const npy_intp n = PyArray_DIM(t, 0);
    out.time.resize(static_cast<size_t>(n));
    out.position.resize(static_cast<size_t>(n));
    if (v) out.velocity.resize(static_cast<size_t>(n));
    const double* td = static_cast<const double*>(PyArray_DATA(t));
    const double* pd = static_cast<const double*>(PyArray_DATA(p));
    const double* vd = v ? static_cast<const double*>(PyArray_DATA(v)) : nullptr;
    for (npy_intp i = 0; i < n; ++i) {
        out.time[static_cast<size_t>(i)] = td[i];
        out.position[static_cast<size_t>(i)] = {pd[3 * i], pd[3 * i + 1], pd[3 * i + 2]};
        if (vd) out.velocity[static_cast<size_t>(i)] = {vd[3 * i], vd[3 * i + 1], vd[3 * i + 2]};
    }
    out.has_velocity = vd != nullptr;
    Py_DECREF(t); Py_DECREF(p); Py_XDECREF(v); Py_DECREF(seq);
    try {
        out.validate(name);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return false;
    }
    return true;
}

ObserverConvention parse_convention(const char* value) {
    const std::string mode(value ? value : "earth_geocentric_offset");
    if (mode == "earth_geocentric_offset" || mode == "vbm") return ObserverConvention::EarthGeocentricOffset;
    if (mode == "heliocentric_observer") return ObserverConvention::HeliocentricObserver;
    if (mode == "gulls") return ObserverConvention::Gulls;
    throw std::invalid_argument("observer_convention must be earth_geocentric_offset, heliocentric_observer, or gulls");
}

PyObject* make_array(const std::vector<double>& values) {
    npy_intp dims[1] = {static_cast<npy_intp>(values.size())};
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
    if (!out) return nullptr;
    std::copy(values.begin(), values.end(), static_cast<double*>(PyArray_DATA(out)));
    return reinterpret_cast<PyObject*>(out);
}

PyObject* make_trajectory_dict(const std::vector<TrajectoryComponents>& rows) {
    PyObject* dict = PyDict_New();
    if (!dict) return nullptr;
    struct Column {
        const char* name;
        double TrajectoryComponents::*member;
    };
    constexpr Column columns[] = {
        {"tau", &TrajectoryComponents::tau}, {"beta", &TrajectoryComponents::beta},
        {"u", &TrajectoryComponents::u}, {"earth_n", &TrajectoryComponents::earth_n},
        {"earth_e", &TrajectoryComponents::earth_e}, {"satellite_n", &TrajectoryComponents::satellite_n},
        {"satellite_e", &TrajectoryComponents::satellite_e}, {"observer_n", &TrajectoryComponents::observer_n},
        {"observer_e", &TrajectoryComponents::observer_e},
    };
    npy_intp dims[] = {static_cast<npy_intp>(rows.size())};
    for (const auto& column : columns) {
        PyObject* array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (array) {
            auto* values = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));
            for (size_t i = 0; i < rows.size(); ++i)
                values[i] = rows[i].*column.member;
        }
        if (!array || PyDict_SetItemString(dict, column.name, array) < 0) {
            Py_XDECREF(array); Py_DECREF(dict); return nullptr;
        }
        Py_DECREF(array);
    }
    return dict;
}

struct GroupProfile {
    double weight=0, mean_a=0, mean_flux=0, xx=0, xy=0, fs=0, fb=0;
};

struct EvaluatorData {
    std::unique_ptr<ParallaxTrajectory> trajectory;
    std::unique_ptr<VbmMagnification> magnification;
    std::vector<double> time;
    std::vector<double> flux;
    std::vector<double> ferr;
    std::vector<long long> dataset;
    bool finite_source = false;
    int nparam = 0;
    // Epoch-only trajectory geometry, computed once per evaluator.
    std::vector<ParallaxTrajectory::SiteGeometry> geometry;
    // Magnifications of the most recent evaluation, so the Jacobian requested
    // at the same parameters (the optimizer's usual pattern) skips the
    // expensive magnification calls.
    std::vector<double> cached_raw;
    std::vector<double> cached_magnifications;
    std::vector<double> tau, beta, separation, d_du, d_drho, weights;
    std::vector<unsigned char> profile_mask;
    std::vector<GroupProfile> groups;
    bool profile_valid=false;
};

struct EvaluatorObject {
    PyObject_HEAD
    EvaluatorData data;
};

bool parse_raw(EvaluatorObject* self, PyObject* obj, std::vector<double>& raw) {
    PyArrayObject* array = as_double_array(obj);
    if (!array) return false;
    if (PyArray_NDIM(array) != 1 || PyArray_DIM(array, 0) != self->data.nparam) {
        PyErr_Format(PyExc_ValueError, "raw_params must be a one-dimensional array of length %d", self->data.nparam);
        Py_DECREF(array); return false;
    }
    const double* data = static_cast<const double*>(PyArray_DATA(array));
    raw.assign(data, data + self->data.nparam);
    Py_DECREF(array);
    for (double value : raw) if (!std::isfinite(value)) { PyErr_SetString(PyExc_ValueError, "raw_params must be finite"); return false; }
    return true;
}

void unpack_raw(EvaluatorObject* self, const std::vector<double>& raw, double& t0, double& tE, double& u0, double& rho, double& piEN, double& piEE) {
    t0 = raw[0];
    tE = std::exp(raw[1]);
    u0 = raw[2];
    int pi_index = 3;
    rho = 0.0;
    if (self->data.finite_source) { rho = singlelensfitter::espl_radius_from_log(raw[3]); pi_index = 4; }
    piEN = raw[pi_index]; piEE = raw[pi_index + 1];
    if (!(tE > 0.0) || (self->data.finite_source && !(rho > 0.0))) throw std::invalid_argument("log_tE/log_rho produced a non-positive physical parameter");
}

bool active_mask(PyObject* obj, size_t n, std::vector<unsigned char>& mask) {
    mask.assign(n, 1);
    if (is_none(obj)) return true;
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY));
    if (!array) return false;
    if (PyArray_NDIM(array) != 1 || static_cast<size_t>(PyArray_DIM(array, 0)) != n) {
        PyErr_SetString(PyExc_ValueError, "active_mask must match the data length");
        Py_DECREF(array); return false;
    }
    const npy_bool* data = static_cast<const npy_bool*>(PyArray_DATA(array));
    for (size_t i = 0; i < n; ++i) mask[i] = data[i] ? 1 : 0;
    Py_DECREF(array); return true;
}

// Geometry and brightness are independent of the active photometry mask.
void prepare_curve(EvaluatorObject* self, const std::vector<double>& raw) {
    if (self->data.cached_raw == raw) return;
    self->data.cached_raw.clear();
    self->data.profile_valid=false;
    double t0, te, u0, rho, north, east;
    unpack_raw(self,raw,t0,te,u0,rho,north,east);
    if (self->data.finite_source && rho > singlelensfitter::max_espl_rho)
        throw std::invalid_argument("ESPL requires rho <= 100");
    const auto n=self->data.time.size();
    self->data.tau.resize(n); self->data.beta.resize(n); self->data.separation.resize(n);
    self->data.cached_magnifications.resize(n); self->data.d_du.resize(n); self->data.d_drho.resize(n);
    const double peak=self->data.trajectory->source_peak_time(t0);
    for (size_t i=0; i<n; ++i) {
        const auto row=self->data.trajectory->at(self->data.geometry[i],peak,te,u0,north,east);
        self->data.tau[i]=row.tau; self->data.beta[i]=row.beta; self->data.separation[i]=row.u;
    }
    if (self->data.finite_source) {
        self->data.magnification->batch<true,false>(self->data.separation.data(),n,rho,
            self->data.cached_magnifications.data(),self->data.d_du.data(),self->data.d_drho.data());
    } else {
        for (size_t i=0; i<n; ++i)
            self->data.magnification->value_and_gradient(self->data.separation[i],rho,
                self->data.cached_magnifications[i],self->data.d_du[i],self->data.d_drho[i]);
    }
    self->data.cached_raw=raw;
}

void prepare_profile(EvaluatorData& data, const std::vector<unsigned char>& mask) {
    if (data.profile_valid && data.profile_mask == mask) return;
    data.profile_valid=false;
    std::fill(data.groups.begin(),data.groups.end(),GroupProfile{});
    for (size_t i=0; i<mask.size(); ++i) if (mask[i]) {
        auto& g=data.groups[data.dataset[i]];
        const double w=data.weights[i];
        g.weight+=w; g.mean_a+=w*data.cached_magnifications[i]; g.mean_flux+=w*data.flux[i];
    }
    for (auto& g : data.groups) if (g.weight>0) {
        g.mean_a/=g.weight; g.mean_flux/=g.weight;
    }
    for (size_t i=0; i<mask.size(); ++i) if (mask[i]) {
        auto& g=data.groups[data.dataset[i]];
        const double dx=data.cached_magnifications[i]-g.mean_a, wx=data.weights[i]*dx;
        g.xx+=wx*dx; g.xy+=wx*(data.flux[i]-g.mean_flux);
    }
    for (auto& g : data.groups) {
        if (!std::isfinite(g.mean_a) || !std::isfinite(g.mean_flux) ||
            !std::isfinite(g.xx) || !std::isfinite(g.xy))
            throw std::invalid_argument("dataset flux profile is non-finite");
        g.fs=g.xx>0 ? g.xy/g.xx : 0.;
        g.fb=g.mean_flux-g.fs*g.mean_a;
    }
    data.profile_mask=mask;
    data.profile_valid=true;
}

bool evaluate_internal(EvaluatorObject* self, const std::vector<double>& raw, std::vector<double>& residual, PyObject* mask_obj, bool return_model=false) {
    try {
        prepare_curve(self,raw);
        const size_t n = self->data.time.size();
        std::vector<unsigned char> mask;
        if (!active_mask(mask_obj, n, mask)) return false;
        prepare_profile(self->data,mask);
        residual.resize(n);
        for (size_t i=0; i<n; ++i) {
            const auto& g=self->data.groups[self->data.dataset[i]];
            const double model=g.fs*self->data.cached_magnifications[i]+g.fb;
            residual[i] = return_model ? model :
                (mask[i] ? (self->data.flux[i]-model)/self->data.ferr[i] : 0.0);
            if (!std::isfinite(residual[i])) throw std::invalid_argument("profiled residual is not finite");
        }
        return true;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return false;
    }
}

PyObject* evaluator_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self=reinterpret_cast<EvaluatorObject*>(type->tp_alloc(type,0));
    if (self) new (&self->data) EvaluatorData{};
    return reinterpret_cast<PyObject*>(self);
}

int evaluator_init(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *time_obj = nullptr, *flux_obj = nullptr, *ferr_obj = nullptr, *dataset_obj = nullptr;
    double ra = 0.0, dec = 0.0, tref = 0.0;
    const char* time_kind = "jd";
    const char* convention = "earth_geocentric_offset";
    PyObject *earth_obj = Py_None, *sat_obj = Py_None, *observer_obj = Py_None, *reference_obj = Py_None;
    int finite_source = 0;
    PyObject* table_obj = Py_None;
    double tol = 1e-4, reltol = 1e-4;
    int allow_extrapolation = 0;
    double a = 0;
    static const char* kwlist[] = {
        "time", "flux", "ferr", "dataset_id", "ra_deg", "dec_deg", "tref", "time_kind", "observer_convention",
        "earth_ephemeris", "satellite_or_observer_ephemeris", "reference_ephemeris", "finite_source", "espl_table_path",
        "magnification_tol", "magnification_reltol", "allow_extrapolation", "linear_limb_darkening", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOdddss|OOOpOddpd", const_cast<char**>(kwlist),
            &time_obj, &flux_obj, &ferr_obj, &dataset_obj, &ra, &dec, &tref, &time_kind, &convention,
            &earth_obj, &sat_obj, &reference_obj, &finite_source, &table_obj, &tol, &reltol, &allow_extrapolation,&a)) return -1;
    PyArrayObject* time_arr = as_double_array(time_obj);
    PyArrayObject* flux_arr = as_double_array(flux_obj);
    PyArrayObject* ferr_arr = as_double_array(ferr_obj);
    PyArrayObject* dataset_arr = is_none(dataset_obj) ? nullptr : as_int_array(dataset_obj);
    if (!time_arr || !flux_arr || !ferr_arr || (!is_none(dataset_obj) && !dataset_arr)) goto fail;
    if (PyArray_NDIM(time_arr) != 1 || PyArray_NDIM(flux_arr) != 1 || PyArray_NDIM(ferr_arr) != 1 ||
        PyArray_DIM(time_arr, 0) != PyArray_DIM(flux_arr, 0) || PyArray_DIM(time_arr, 0) != PyArray_DIM(ferr_arr, 0) ||
        (dataset_arr && (PyArray_NDIM(dataset_arr) != 1 || PyArray_DIM(dataset_arr, 0) != PyArray_DIM(time_arr, 0)))) {
        PyErr_SetString(PyExc_ValueError, "time, flux, ferr, and dataset_id must be one-dimensional and have equal length"); goto fail;
    }
    {
        const npy_intp n = PyArray_DIM(time_arr, 0);
        const double* td = static_cast<const double*>(PyArray_DATA(time_arr));
        const double* fd = static_cast<const double*>(PyArray_DATA(flux_arr));
        const double* ed = static_cast<const double*>(PyArray_DATA(ferr_arr));
        self->data.time.assign(td, td + n); self->data.flux.assign(fd, fd + n); self->data.ferr.assign(ed, ed + n);
        self->data.dataset.assign(static_cast<size_t>(n), 0);
        if (dataset_arr) {
            const long long* gd = static_cast<const long long*>(PyArray_DATA(dataset_arr));
            for (npy_intp i = 0; i < n; ++i) self->data.dataset[static_cast<size_t>(i)] = gd[i];
        }
        for (npy_intp i = 0; i < n; ++i) if (!std::isfinite(self->data.time[i]) || !std::isfinite(self->data.flux[i]) || !(self->data.ferr[i] > 0.0) || !std::isfinite(self->data.ferr[i]) || self->data.dataset[static_cast<size_t>(i)] < 0) {
            PyErr_SetString(PyExc_ValueError, "time/flux must be finite, ferr must be positive, and dataset_id non-negative"); goto fail;
        }
    }
    try {
        size_t groups=1;
        self->data.weights.resize(self->data.ferr.size());
        for (size_t i=0; i<self->data.ferr.size(); ++i) {
            if (self->data.dataset[i]>1000000) throw std::invalid_argument("dataset_id contains an invalid group id");
            groups=std::max(groups,static_cast<size_t>(self->data.dataset[i])+1);
            self->data.weights[i]=1/(self->data.ferr[i]*self->data.ferr[i]);
        }
        self->data.groups.resize(groups);
        self->data.profile_valid=false;
        Ephemeris earth, satellite, observer, reference;
        const bool has_earth = read_ephemeris(earth_obj, "earth_ephemeris", earth);
        const bool has_satellite = read_ephemeris(sat_obj, "satellite_or_observer_ephemeris", satellite);
        const bool has_observer = read_ephemeris(observer_obj, "satellite_or_observer_ephemeris", observer);
        const bool has_reference = read_ephemeris(reference_obj, "reference_ephemeris", reference);
        earth.allow_extrapolation = allow_extrapolation != 0;
        satellite.allow_extrapolation = allow_extrapolation != 0;
        observer.allow_extrapolation = allow_extrapolation != 0;
        reference.allow_extrapolation = allow_extrapolation != 0;
        const ObserverConvention mode = parse_convention(convention);
        if (mode == ObserverConvention::EarthGeocentricOffset && !has_earth) throw std::invalid_argument("earth_geocentric_offset requires earth_ephemeris");
        if (mode != ObserverConvention::EarthGeocentricOffset && ((!has_satellite && !has_observer) || !has_reference)) {
            throw std::invalid_argument("heliocentric_observer/gulls require observer and reference ephemerides");
        }
        // For the public API, satellite_or_observer_ephemeris is the observer
        // table in heliocentric/gulls modes and the geocentric offset in the
        // earth mode.
        if (mode != ObserverConvention::EarthGeocentricOffset) {
            observer = has_observer ? observer : satellite;
        }
        std::string table;
        if (!is_none(table_obj)) {
            PyObject* encoded = PyObject_Str(table_obj);
            if (!encoded) goto fail;
            const char* value = PyUnicode_AsUTF8(encoded);
            if (value) table = value;
            Py_DECREF(encoded);
        }
        self->data.trajectory = std::make_unique<ParallaxTrajectory>(
            std::move(earth), std::move(satellite), has_satellite, std::move(observer), has_observer || has_satellite,
            std::move(reference), has_reference, ra, dec, tref, mode, std::string(time_kind), mode == ObserverConvention::Gulls);
        self->data.geometry.clear(); self->data.geometry.reserve(self->data.time.size());
        for (double t : self->data.time) self->data.geometry.push_back(self->data.trajectory->site_geometry(t));
        self->data.cached_raw.clear(); self->data.cached_magnifications.clear();
        self->data.finite_source = finite_source != 0;
        self->data.nparam = self->data.finite_source ? 6 : 5;
        self->data.magnification = std::make_unique<VbmMagnification>(self->data.finite_source, table, tol, reltol, a);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what()); goto fail;
    }
    Py_DECREF(time_arr); Py_DECREF(flux_arr); Py_DECREF(ferr_arr); Py_XDECREF(dataset_arr);
    return 0;
fail:
    Py_XDECREF(time_arr); Py_XDECREF(flux_arr); Py_XDECREF(ferr_arr); Py_XDECREF(dataset_arr);
    return -1;
}

void evaluator_dealloc(EvaluatorObject* self) {
    self->data.~EvaluatorData(); Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* evaluator_trajectory(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* raw_obj = nullptr; int components = 0;
    static const char* kwlist[] = {"raw_params", "components", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", const_cast<char**>(kwlist), &raw_obj, &components)) return nullptr;
    std::vector<double> raw; if (!parse_raw(self, raw_obj, raw)) return nullptr;
    try {
        double t0, tE, u0, rho, piEN, piEE; unpack_raw(self, raw, t0, tE, u0, rho, piEN, piEE); (void)rho;
        if (components) {
            std::vector<TrajectoryComponents> rows;
            rows.reserve(self->data.time.size());
            for (double t : self->data.time)
                rows.push_back(self->data.trajectory->at(t, t0, tE, u0, piEN, piEE));
            return make_trajectory_dict(rows);
        }
        if (self->data.cached_raw == raw)
            return Py_BuildValue("NNN", make_array(self->data.tau), make_array(self->data.beta),
                                 make_array(self->data.separation));
        const auto n = self->data.geometry.size();
        std::vector<double> tau(n), beta(n), separation(n);
        const double peak = self->data.trajectory->source_peak_time(t0);
        for (size_t i = 0; i < n; ++i) {
            const auto row = self->data.trajectory->at(self->data.geometry[i], peak, tE, u0, piEN, piEE);
            tau[i] = row.tau;
            beta[i] = row.beta;
            separation[i] = row.u;
        }
        return Py_BuildValue("NNN", make_array(tau), make_array(beta), make_array(separation));
    } catch (const std::exception& exc) { PyErr_SetString(PyExc_ValueError, exc.what()); return nullptr; }
}

PyObject* evaluator_evaluate(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* raw_obj = nullptr; PyObject* mask_obj = Py_None;
    static const char* kwlist[] = {"raw_params", "active_mask", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", const_cast<char**>(kwlist), &raw_obj, &mask_obj)) return nullptr;
    std::vector<double> raw, model;
    if (!parse_raw(self,raw_obj,raw)) return nullptr;
    // evaluate() always returns the full-data profile; validate a supplied mask.
    std::vector<unsigned char> mask;
    if (!is_none(mask_obj) && !active_mask(mask_obj,self->data.time.size(),mask)) return nullptr;
    if (!evaluate_internal(self,raw,model,Py_None,true)) return nullptr;
    return make_array(model);
}

PyObject* evaluator_magnification(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* raw_obj = nullptr;
    static const char* kwlist[] = {"raw_params", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", const_cast<char**>(kwlist), &raw_obj)) return nullptr;
    std::vector<double> raw; if (!parse_raw(self, raw_obj, raw)) return nullptr;
    try { prepare_curve(self,raw); return make_array(self->data.cached_magnifications); }
    catch(const std::exception& exc){PyErr_SetString(PyExc_ValueError,exc.what());return nullptr;}
}

PyObject* evaluator_residual(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* raw_obj=nullptr; PyObject* mask_obj=Py_None; static const char* kwlist[]={"raw_params","active_mask",nullptr};
    if(!PyArg_ParseTupleAndKeywords(args,kwargs,"O|O",const_cast<char**>(kwlist),&raw_obj,&mask_obj))return nullptr;
    std::vector<double> raw,residual; if(!parse_raw(self,raw_obj,raw))return nullptr; if(!evaluate_internal(self,raw,residual,mask_obj))return nullptr; return make_array(residual);
}

PyObject* evaluator_jacobian_fd(EvaluatorObject* self, PyObject* raw_obj, PyObject* mask_obj, double fd_step) {
    std::vector<double> raw,base; if(!parse_raw(self,raw_obj,raw))return nullptr; if(!evaluate_internal(self,raw,base,mask_obj))return nullptr;
    const size_t n=self->data.time.size(); npy_intp dims[2]={static_cast<npy_intp>(n),static_cast<npy_intp>(self->data.nparam)}; PyArrayObject* out=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2,dims,NPY_DOUBLE,0)); if(!out)return nullptr; double* jd=static_cast<double*>(PyArray_DATA(out));
    for(int k=0;k<self->data.nparam;++k){ double step=std::max(std::abs(fd_step),1e-8); if(k==0)step=std::max(step,1e-6); if(k==1|| (self->data.finite_source&&k==3))step=std::max(step,1e-5); std::vector<double> plus=raw,minus=raw,rp,rm; plus[k]+=step;minus[k]-=step; bool okp=evaluate_internal(self,plus,rp,mask_obj); bool okm=evaluate_internal(self,minus,rm,mask_obj); if(okp&&okm){for(size_t i=0;i<n;++i)jd[i*self->data.nparam+k]=(rp[i]-rm[i])/(2*step);} else if(okp){std::vector<double> r0;evaluate_internal(self,raw,r0,mask_obj);for(size_t i=0;i<n;++i)jd[i*self->data.nparam+k]=(rp[i]-r0[i])/step;} else if(okm){std::vector<double> r0;evaluate_internal(self,raw,r0,mask_obj);for(size_t i=0;i<n;++i)jd[i*self->data.nparam+k]=(r0[i]-rm[i])/step;} }
    return reinterpret_cast<PyObject*>(out);
}

inline void magnification_partials(const EvaluatorData& data, size_t i, double tE,
                                   double rho, double piEN, double piEE,
                                   double peak_slope, double* out) {
    const auto& geometry = data.geometry[i];
    const double tau = data.tau[i], beta = data.beta[i];
    const double slope = data.separation[i] > 0. ? data.d_du[i] / data.separation[i] : 0.;
    const double along = slope * tau, across = slope * beta;
    const size_t pi_index = data.finite_source ? 4 : 3;
    out[0] = -along * peak_slope / tE;
    out[1] = -along * (tau + piEN * geometry.d_n + piEE * geometry.d_e);
    out[2] = across;
    if (data.finite_source) out[3] = data.d_drho[i] * rho;
    out[pi_index] = -along * geometry.d_n - across * geometry.d_e;
    out[pi_index + 1] = -along * geometry.d_e + across * geometry.d_n;
}

// Analytic Jacobian of the profiled residual.
//
// The magnification depends on the parameters only through u(t) (and rho), so
// dA/dtheta = A_u du/dtheta + A_rho drho/dtheta, with du/dtheta from the
// trajectory.  The per-dataset flux solution (fs, fb) is the weighted
// regression of flux on A; its exact derivative is
//   dfs = sum_j w_j dA_j (y_j - ym - 2 fs a_j) / Sxx,   a_j = A_j - xm
//   dr_i = -(dfs a_i + fs (dA_i - dxm)) / sigma_i,      dxm = sum_j w_j dA_j / sum_j w_j
// which is what the finite-difference Jacobian approximates by refitting.
bool analytic_jacobian(EvaluatorObject* self, const std::vector<double>& raw, PyObject* mask_obj, double* jac) {
    try {
        double t0, tE, u0, rho, piEN, piEE;
        unpack_raw(self, raw, t0, tE, u0, rho, piEN, piEE);
        const size_t n = self->data.time.size();
        const size_t np = static_cast<size_t>(self->data.nparam);
        std::vector<unsigned char> mask;
        if (!active_mask(mask_obj, n, mask)) return false;
        const double peak_slope = self->data.trajectory->source_peak_time_derivative(t0);
        prepare_curve(self,raw);
        const auto& mags=self->data.cached_magnifications;
        prepare_profile(self->data,mask);
        for (size_t i=0; i<n; ++i) {
            magnification_partials(self->data, i, tE, rho, piEN, piEE, peak_slope, &jac[i*np]);
        }
        const auto& groups=self->data.groups;
        const size_t ng=groups.size();
        std::vector<double> s_dxm(ng * np, 0.0), s_dfs(ng * np, 0.0);  // sum w dA, sum w dA c
        for (size_t i = 0; i < n; ++i) {
            const size_t g = static_cast<size_t>(self->data.dataset[i]);
            if (!(groups[g].xx > 0.0)) continue;
            const double w = mask[i] ? self->data.weights[i] : 0.;
            if (w == 0.0) continue;
            const double c = (self->data.flux[i] - groups[g].mean_flux) - 2.0 * groups[g].fs * (mags[i] - groups[g].mean_a);
            for (size_t k = 0; k < np; ++k) {
                const double wd = w * jac[i * np + k];
                s_dxm[g * np + k] += wd; s_dfs[g * np + k] += wd * c;
            }
        }
        for (size_t g=0; g<ng; ++g) if (groups[g].xx>0.) {
            for (size_t k=0; k<np; ++k) {
                s_dfs[g*np+k] /= groups[g].xx; s_dxm[g*np+k] /= groups[g].weight;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            const size_t g = static_cast<size_t>(self->data.dataset[i]);
            if (!mask[i] || !(groups[g].xx > 0.0)) {
                std::fill_n(&jac[i*np],np,0.); continue;
            }
            const double a = mags[i] - groups[g].mean_a;
            for (size_t k = 0; k < np; ++k) {
                const double dfs = s_dfs[g * np + k];
                const double dxm = s_dxm[g * np + k];
                const double value = -(dfs * a + groups[g].fs * (jac[i * np + k] - dxm)) / self->data.ferr[i];
                if (!std::isfinite(value)) throw std::invalid_argument("analytic Jacobian is not finite");
                jac[i * np + k] = value;
            }
        }
        return true;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return false;
    }
}

PyObject* make_analytic_jacobian(EvaluatorObject* self, const std::vector<double>& raw, PyObject* mask_obj) {
    const size_t n=self->data.time.size(); npy_intp dims[2]={static_cast<npy_intp>(n),static_cast<npy_intp>(self->data.nparam)};
    PyArrayObject* out=reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2,dims,NPY_DOUBLE)); if(!out)return nullptr;
    if (!analytic_jacobian(self,raw,mask_obj,static_cast<double*>(PyArray_DATA(out)))) {
        Py_DECREF(out); return nullptr;
    }
    return reinterpret_cast<PyObject*>(out);
}

PyObject* evaluator_jacobian(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* raw_obj=nullptr; PyObject* mask_obj=Py_None; double fd_step=1e-5; const char* method="analytic"; static const char* kwlist[]={"raw_params","active_mask","fd_step","method",nullptr};
    if(!PyArg_ParseTupleAndKeywords(args,kwargs,"O|Ods",const_cast<char**>(kwlist),&raw_obj,&mask_obj,&fd_step,&method))return nullptr;
    const std::string kind(method);
    if (kind == "fd") return evaluator_jacobian_fd(self, raw_obj, mask_obj, fd_step);
    if (kind != "analytic") { PyErr_SetString(PyExc_ValueError, "method must be 'analytic' or 'fd'"); return nullptr; }
    std::vector<double> raw; if(!parse_raw(self,raw_obj,raw))return nullptr;
    return make_analytic_jacobian(self,raw,mask_obj);
}

PyObject* evaluator_residual_and_jacobian(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *raw_obj=nullptr, *mask_obj=Py_None;
    double fd_step=1e-5;
    static const char* kwlist[]={"raw_params","active_mask","fd_step",nullptr};
    if (!PyArg_ParseTupleAndKeywords(args,kwargs,"O|Od",const_cast<char**>(kwlist),
                                     &raw_obj,&mask_obj,&fd_step)) return nullptr;
    std::vector<double> raw,residual;
    if (!parse_raw(self,raw_obj,raw) || !evaluate_internal(self,raw,residual,mask_obj)) return nullptr;
    PyObject* jac=make_analytic_jacobian(self,raw,mask_obj);
    if (!jac) return nullptr;
    return Py_BuildValue("NN",make_array(residual),jac);
}

PyObject* evaluator_chi2_and_grad(EvaluatorObject* self, PyObject* args) {
    PyObject *params, *active = Py_None;
    if (!PyArg_ParseTuple(args, "O|O", &params, &active)) return nullptr;
    auto& data = self->data;
    std::vector<double> raw;
    std::vector<unsigned char> mask;
    if (!parse_raw(self, params, raw) || !active_mask(active, data.time.size(), mask))
        return nullptr;
    try {
        prepare_curve(self, raw);
        prepare_profile(data, mask);
        double t0, tE, u0, rho, north, east;
        unpack_raw(self, raw, t0, tE, u0, rho, north, east);
        const double peak_slope = data.trajectory->source_peak_time_derivative(t0);
        std::vector<double> grad(data.nparam, 0.);
        double chi2 = 0.;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (!mask[i]) continue;
            const auto& group = data.groups[data.dataset[i]];
            const double residual = data.flux[i] - (group.fs * data.cached_magnifications[i] + group.fb);
            const double weighted = data.weights[i] * residual;
            chi2 += residual * weighted;
            double partials[6];
            magnification_partials(data, i, tE, rho, north, east, peak_slope, partials);
            const double factor = -2. * group.fs * weighted;
            for (size_t j = 0; j < grad.size(); ++j) grad[j] += factor * partials[j];
        }
        return Py_BuildValue("dN", chi2, make_array(grad));
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyMethodDef evaluator_methods[] = {
    {"chi2_and_grad", reinterpret_cast<PyCFunction>(evaluator_chi2_and_grad), METH_VARARGS, "Profiled chi-square and gradient in native coordinates."},
    {"trajectory", reinterpret_cast<PyCFunction>(evaluator_trajectory), METH_VARARGS|METH_KEYWORDS, "Evaluate tau/beta/u and optional trajectory components."},
    {"magnification", reinterpret_cast<PyCFunction>(evaluator_magnification), METH_VARARGS|METH_KEYWORDS, "Evaluate compiled finite-source magnification."},
    {"evaluate", reinterpret_cast<PyCFunction>(evaluator_evaluate), METH_VARARGS|METH_KEYWORDS, "Evaluate profiled model flux."},
    {"residual", reinterpret_cast<PyCFunction>(evaluator_residual), METH_VARARGS|METH_KEYWORDS, "Evaluate profiled weighted residual."},
    {"jacobian", reinterpret_cast<PyCFunction>(evaluator_jacobian), METH_VARARGS|METH_KEYWORDS, "Evaluate adaptive central finite-difference Jacobian."},
    {"residual_and_jacobian", reinterpret_cast<PyCFunction>(evaluator_residual_and_jacobian), METH_VARARGS|METH_KEYWORDS, "Evaluate residual and Jacobian."},
    {nullptr,nullptr,0,nullptr}
};

PyTypeObject EvaluatorType = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_parallax_cpp", "Native parallax backend", -1, nullptr, nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__parallax_cpp(void) {
    import_array();
    EvaluatorType.tp_name = "singlelensfitter._parallax_cpp.ParallaxEvaluator";
    EvaluatorType.tp_basicsize = sizeof(EvaluatorObject);
    EvaluatorType.tp_dealloc = reinterpret_cast<destructor>(evaluator_dealloc);
    EvaluatorType.tp_flags = Py_TPFLAGS_DEFAULT;
    EvaluatorType.tp_doc = "Compiled parallax trajectory and magnification evaluator";
    EvaluatorType.tp_methods = evaluator_methods;
    EvaluatorType.tp_new = evaluator_new;
    EvaluatorType.tp_init = reinterpret_cast<initproc>(evaluator_init);
    if (PyType_Ready(&EvaluatorType) < 0) return nullptr;
    PyObject* mod = PyModule_Create(&module); if (!mod) return nullptr;
    Py_INCREF(&EvaluatorType); if (PyModule_AddObject(mod, "ParallaxEvaluator", reinterpret_cast<PyObject*>(&EvaluatorType)) < 0) {Py_DECREF(&EvaluatorType);Py_DECREF(mod);return nullptr;}
    return mod;
}
