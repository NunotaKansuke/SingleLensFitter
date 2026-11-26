#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>

#include "VBMicrolensingLibrary.h"
#include "cpp/vbm_magnification.hpp"

#ifdef SINGLELENSFITTER_XLA_FFI
#include "xla/ffi/api/ffi.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

PyArrayObject* as_double_array(PyObject* object) {
    return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)
    );
}

// Keep one evaluator per calling thread. Recheck the table on each call so
// switching or updating a file also refreshes the private interpolation copy.
singlelensfitter::VbmMagnification& batch_evaluator(const std::string& path,
                                                    double tol, double reltol, double a) {
    using singlelensfitter::VbmMagnification;
    struct Cache {
        std::string path;
        std::filesystem::file_time_type modified;
        std::unique_ptr<VbmMagnification> evaluator;
    };
    static thread_local Cache last{};
    std::error_code error;
    const auto modified = std::filesystem::last_write_time(path, error);
    if (!last.evaluator || error || last.path != path || last.modified != modified ||
        !std::filesystem::path(path).is_absolute()) {
        last = {path, modified,
                std::make_unique<VbmMagnification>(true, path, tol, reltol, a)};
    } else {
        last.evaluator->configure(tol, reltol, a);
    }
    return *last.evaluator;
}

PyObject* finite_source_values(PyObject* args, PyObject* kwargs, bool light_curve) {
    PyObject* u_object = nullptr;
    PyObject* table_object = Py_None;
    double rho = 0.0;
    double tol = 1.0e-4;
    double reltol = 1.0e-4, a = 0.;
    static const char* kwlist[] = {
        "u", "rho", "espl_table", "tol", "reltol", "linear_limb_darkening", nullptr
    };
    double t0 = 0, tE = 1, u0 = 0;
    if (light_curve) {
        static const char* curve_keys[] = {"time", "rho", "t0", "tE", "u0", "espl_table",
                                          "tol", "reltol", "linear_limb_darkening", nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Odddd|Oddd", const_cast<char**>(curve_keys),
                &u_object, &rho, &t0, &tE, &u0, &table_object, &tol, &reltol, &a))
            return nullptr;
    } else if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Od|Oddd", const_cast<char**>(kwlist),
                   &u_object, &rho, &table_object, &tol, &reltol, &a)) {
        return nullptr;
    }

    PyArrayObject* u_array = as_double_array(u_object);
    if (u_array == nullptr) return nullptr;
    if (
        PyArray_NDIM(u_array) != 1
        || !std::isfinite(rho)
        || !(rho > 0.0) || rho > singlelensfitter::max_espl_rho
        || !std::isfinite(tol)
        || !(tol > 0.0)
        || !std::isfinite(reltol)
        || !(reltol > 0.0)
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "u must be one-dimensional; rho, tol, and reltol must be finite and positive, with rho <= 100."
        );
        Py_DECREF(u_array);
        return nullptr;
    }

    const char* table = nullptr;
    if (table_object != Py_None) {
        table = PyUnicode_AsUTF8(table_object);
        if (table == nullptr) {
            Py_DECREF(u_array);
            return nullptr;
        }
    }

    const npy_intp n = PyArray_DIM(u_array, 0);
    npy_intp dimensions[1] = {n};
    auto* output = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, dimensions, NPY_DOUBLE)
    );
    if (output == nullptr) {
        Py_DECREF(u_array);
        return nullptr;
    }

    try {
        auto& magnifier = batch_evaluator(table ? table : "", tol, reltol, a);

        const double* input = static_cast<const double*>(
            PyArray_DATA(u_array)
        );
        double* values = static_cast<double*>(PyArray_DATA(output));
        if (light_curve) {
            const double duration = std::max(std::abs(tE), 1e-12);
            for (npy_intp i = 0; i < n; ++i) {
                const double tau = (input[i] - t0) / duration;
                values[i] = std::sqrt(tau*tau + u0*u0);
            }
            input = values;
        }
        magnifier.batch<false,false>(input,n,rho,values);
    } catch (const std::exception& exception) {
        Py_DECREF(u_array);
        Py_DECREF(output);
        PyErr_SetString(PyExc_ValueError, exception.what());
        return nullptr;
    }

    Py_DECREF(u_array);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* finite_source_magnification(PyObject*, PyObject* args, PyObject* kwargs) {
    return finite_source_values(args, kwargs, false);
}

PyObject* finite_source_light_curve(PyObject*, PyObject* args, PyObject* kwargs) {
    return finite_source_values(args, kwargs, true);
}

PyObject* finite_source_magnification_gradient(
    PyObject*, PyObject* args, PyObject* kwargs
) {
    PyObject* u_object = nullptr;
    PyObject* table_object = Py_None;
    double rho = 0.0;
    double tol = 1.0e-4;
    double reltol = 1.0e-4, a = 0.;
    int coefficient_gradient = 0;
    static const char* kwlist[] = {
        "u", "rho", "espl_table", "tol", "reltol", "linear_limb_darkening", "coefficient_gradient", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "Od|Odddp", const_cast<char**>(kwlist),
            &u_object, &rho, &table_object, &tol, &reltol, &a, &coefficient_gradient)) {
        return nullptr;
    }
    PyArrayObject* u_array = as_double_array(u_object);
    if (u_array == nullptr) return nullptr;
    if (PyArray_NDIM(u_array) != 1 || !std::isfinite(rho) || !(rho > 0.0) || rho > singlelensfitter::max_espl_rho
        || !std::isfinite(tol) || !(tol > 0.0)
        || !std::isfinite(reltol) || !(reltol > 0.0)) {
        PyErr_SetString(
            PyExc_ValueError,
            "u must be one-dimensional; rho, tol, and reltol must be finite and positive, with rho <= 100."
        );
        Py_DECREF(u_array);
        return nullptr;
    }
    std::string table;
    if (table_object != Py_None) {
        const char* text = PyUnicode_AsUTF8(table_object);
        if (text == nullptr) {
            Py_DECREF(u_array);
            return nullptr;
        }
        table = text;
    }
    const npy_intp n = PyArray_DIM(u_array, 0);
    npy_intp dimensions[1] = {n};
    std::vector<PyArrayObject*> outputs(coefficient_gradient ? 4 : 3);
    for (auto& output : outputs) {
        output = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dimensions, NPY_DOUBLE));
    }
    if (std::find(outputs.begin(), outputs.end(), nullptr) != outputs.end()) {
        for (auto* output : outputs) Py_XDECREF(output);
        Py_DECREF(u_array);
        return nullptr;
    }
    try {
        auto& magnifier = batch_evaluator(table, tol, reltol, a);
        const double* input = static_cast<const double*>(PyArray_DATA(u_array));
        double* value = static_cast<double*>(PyArray_DATA(outputs[0]));
        double* d_du = static_cast<double*>(PyArray_DATA(outputs[1]));
        double* d_drho = static_cast<double*>(PyArray_DATA(outputs[2]));
        if (coefficient_gradient) {
            double* da = static_cast<double*>(PyArray_DATA(outputs[3]));
            magnifier.batch<true,true>(input,n,rho,value,d_du,d_drho,da);
        } else {
            magnifier.batch<true,false>(input,n,rho,value,d_du,d_drho);
        }
    } catch (const std::exception& exception) {
        for (auto* output : outputs) Py_DECREF(output);
        Py_DECREF(u_array);
        PyErr_SetString(PyExc_ValueError, exception.what());
        return nullptr;
    }
    Py_DECREF(u_array);
    if (coefficient_gradient) return Py_BuildValue("NNNN", outputs[0], outputs[1], outputs[2], outputs[3]);
    return Py_BuildValue("NNN", outputs[0], outputs[1], outputs[2]);
}

PyObject* score_finite_source_seeds(
    PyObject*, PyObject* args, PyObject* kwargs
) {
    PyObject* time_object = nullptr;
    PyObject* flux_object = nullptr;
    PyObject* ferr_object = nullptr;
    PyObject* seeds_object = nullptr;
    PyObject* table_object = Py_None;
    double tol = 1.0e-4;
    double reltol = 1.0e-4;
    static const char* kwlist[] = {
        "time", "flux", "ferr", "seeds", "espl_table", "tol", "reltol", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOO|Odd",
            const_cast<char**>(kwlist),
            &time_object,
            &flux_object,
            &ferr_object,
            &seeds_object,
            &table_object,
            &tol,
            &reltol
        )) {
        return nullptr;
    }

    PyArrayObject* time_array = as_double_array(time_object);
    PyArrayObject* flux_array = as_double_array(flux_object);
    PyArrayObject* ferr_array = as_double_array(ferr_object);
    PyArrayObject* seeds_array = as_double_array(seeds_object);
    if (
        time_array == nullptr
        || flux_array == nullptr
        || ferr_array == nullptr
        || seeds_array == nullptr
    ) {
        Py_XDECREF(time_array);
        Py_XDECREF(flux_array);
        Py_XDECREF(ferr_array);
        Py_XDECREF(seeds_array);
        return nullptr;
    }

    const bool valid_shape = (
        PyArray_NDIM(time_array) == 1
        && PyArray_NDIM(flux_array) == 1
        && PyArray_NDIM(ferr_array) == 1
        && PyArray_NDIM(seeds_array) == 2
        && PyArray_DIM(time_array, 0) == PyArray_DIM(flux_array, 0)
        && PyArray_DIM(time_array, 0) == PyArray_DIM(ferr_array, 0)
        && PyArray_DIM(seeds_array, 1) == 4
        && PyArray_DIM(seeds_array, 0) > 0
        && PyArray_DIM(time_array, 0) > 0
    );
    if (!valid_shape || !std::isfinite(tol) || !(tol > 0.0)
        || !std::isfinite(reltol) || !(reltol > 0.0)) {
        PyErr_SetString(
            PyExc_ValueError,
            "time, flux, and ferr must be equal-length 1-D arrays; "
            "seeds must have shape (n_seeds, 4); tolerances must be positive."
        );
        Py_DECREF(time_array);
        Py_DECREF(flux_array);
        Py_DECREF(ferr_array);
        Py_DECREF(seeds_array);
        return nullptr;
    }

    const char* table = nullptr;
    if (table_object != Py_None) {
        table = PyUnicode_AsUTF8(table_object);
        if (table == nullptr) {
            Py_DECREF(time_array);
            Py_DECREF(flux_array);
            Py_DECREF(ferr_array);
            Py_DECREF(seeds_array);
            return nullptr;
        }
    }

    const npy_intp n_points = PyArray_DIM(time_array, 0);
    const npy_intp n_seeds = PyArray_DIM(seeds_array, 0);
    npy_intp dimensions[1] = {n_seeds};
    auto* output = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, dimensions, NPY_DOUBLE)
    );
    if (output == nullptr) {
        Py_DECREF(time_array);
        Py_DECREF(flux_array);
        Py_DECREF(ferr_array);
        Py_DECREF(seeds_array);
        return nullptr;
    }

    const double* time = static_cast<const double*>(PyArray_DATA(time_array));
    const double* flux = static_cast<const double*>(PyArray_DATA(flux_array));
    const double* ferr = static_cast<const double*>(PyArray_DATA(ferr_array));
    const double* seeds = static_cast<const double*>(PyArray_DATA(seeds_array));
    double* scores = static_cast<double*>(PyArray_DATA(output));

    bool invalid_data = false;
    for (npy_intp index = 0; index < n_points; ++index) {
        if (!std::isfinite(time[index]) || !std::isfinite(flux[index])
            || !std::isfinite(ferr[index]) || !(ferr[index] > 0.0)) {
            invalid_data = true;
            break;
        }
    }
    if (invalid_data) {
        PyErr_SetString(PyExc_ValueError, "time, flux, and ferr must be finite with positive ferr.");
        Py_DECREF(time_array);
        Py_DECREF(flux_array);
        Py_DECREF(ferr_array);
        Py_DECREF(seeds_array);
        Py_DECREF(output);
        return nullptr;
    }

    try {
        singlelensfitter::VbmMagnification magnifier(true, table ? table : "", tol, reltol);

        std::vector<double> magnifications(n_points);
        for (npy_intp seed_index = 0; seed_index < n_seeds; ++seed_index) {
            const double* seed = seeds + seed_index * 4;
            double& score = scores[seed_index];
            score = std::numeric_limits<double>::infinity();
            if (!std::isfinite(seed[0]) || !std::isfinite(seed[1])
                || !std::isfinite(seed[2]) || !std::isfinite(seed[3])
                || seed[1] == 0.0) {
                continue;
            }
            const double rho = singlelensfitter::espl_radius_from_log(std::max(-50.0, seed[3]));
            if (rho > singlelensfitter::max_espl_rho) continue;
            const double tE = std::abs(seed[1]);
            double weight_sum = 0.0;
            double weighted_magnification = 0.0;
            double weighted_flux = 0.0;
            bool invalid_seed = false;
            for (npy_intp index = 0; index < n_points; ++index) {
                const double u = std::sqrt(
                    std::pow((time[index] - seed[0]) / tE, 2.0)
                    + seed[2] * seed[2]
                );
                if (!std::isfinite(u)) {
                    invalid_seed = true;
                    break;
                }
                const double magnification = magnifications[index] = magnifier(u, rho);
                if (!std::isfinite(magnification) || !(magnification > 0.0)) {
                    invalid_seed = true;
                    break;
                }
                const double error = std::max(ferr[index], 1.0e-12);
                const double weight = 1.0 / (error * error);
                weight_sum += weight;
                weighted_magnification += weight * magnification;
                weighted_flux += weight * flux[index];
            }
            if (invalid_seed || !(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
                continue;
            }
            const double mean_magnification = weighted_magnification / weight_sum;
            const double mean_flux = weighted_flux / weight_sum;
            double denominator = 0.0;
            double numerator = 0.0;
            for (npy_intp index = 0; index < n_points; ++index) {
                const double magnification = magnifications[index];
                const double error = std::max(ferr[index], 1.0e-12);
                const double weight = 1.0 / (error * error);
                const double centered_magnification = magnification - mean_magnification;
                const double centered_flux = flux[index] - mean_flux;
                denominator += weight * centered_magnification * centered_magnification;
                numerator += weight * centered_magnification * centered_flux;
            }
            if (!std::isfinite(denominator) || !(denominator > 0.0)) {
                continue;
            }
            const double source_flux = numerator / denominator;
            const double blend_flux = mean_flux - source_flux * mean_magnification;
            if (!std::isfinite(source_flux) || !std::isfinite(blend_flux)) {
                continue;
            }
            double chi2 = 0.0;
            for (npy_intp index = 0; index < n_points; ++index) {
                const double magnification = magnifications[index];
                const double error = std::max(ferr[index], 1.0e-12);
                const double residual = (
                    flux[index] - (source_flux * magnification + blend_flux)
                ) / error;
                chi2 += residual * residual;
            }
            if (std::isfinite(chi2)) {
                score = chi2;
            }
        }
    } catch (const std::exception& exception) {
        Py_DECREF(time_array);
        Py_DECREF(flux_array);
        Py_DECREF(ferr_array);
        Py_DECREF(seeds_array);
        Py_DECREF(output);
        PyErr_SetString(PyExc_ValueError, exception.what());
        return nullptr;
    }

    Py_DECREF(time_array);
    Py_DECREF(flux_array);
    Py_DECREF(ferr_array);
    Py_DECREF(seeds_array);
    return reinterpret_cast<PyObject*>(output);
}

#ifdef SINGLELENSFITTER_XLA_FFI
namespace ffi = xla::ffi;
using Vector = ffi::BufferR1<ffi::F64>;
using Scalar = ffi::BufferR0<ffi::F64>;

ffi::Error fspl_partials(Vector u, Scalar rho, Scalar a, ffi::Result<Vector> out,
                         std::string_view table) {
    const size_t n = u.element_count();
    if (out->element_count() != 4*n || table.empty())
        return ffi::Error::InvalidArgument("Expected four FSPL outputs per separation and an ESPL table.");
    double* values = out->typed_data();
    const double r = *rho.typed_data(), coefficient = *a.typed_data();
    if (!(r > 0 && r <= singlelensfitter::max_espl_rho && coefficient >= 0 && coefficient <= 1) ||
        !std::all_of(u.typed_data(), u.typed_data()+n, [](double x){ return std::isfinite(x); })) {
        std::fill_n(values, 4*n, std::numeric_limits<double>::quiet_NaN());
        return ffi::Error::Success();
    }
    try {
        auto& magnifier = batch_evaluator(std::string(table), 1e-4, 1e-4, coefficient);
        magnifier.batch<true,true>(u.typed_data(), n, r, values, values+n, values+2*n, values+3*n);
    } catch (const std::exception& e) {
        return ffi::Error::InvalidArgument(e.what());
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(fspl_handler, fspl_partials,
    ffi::Ffi::Bind().Arg<Vector>().Arg<Scalar>().Arg<Scalar>().Ret<Vector>().Attr<std::string_view>("table"));

PyObject* fspl_ffi_capsule(PyObject*, PyObject*) {
    return PyCapsule_New(reinterpret_cast<void*>(static_cast<XLA_FFI_Handler*>(fspl_handler)),
                         nullptr, nullptr);
}
#endif

PyMethodDef methods[] = {
#ifdef SINGLELENSFITTER_XLA_FFI
    {"fspl_ffi_capsule", fspl_ffi_capsule, METH_NOARGS, "CPU XLA handler for finite-source partials."},
#endif
    {"fspl_light_curve", reinterpret_cast<PyCFunction>(finite_source_light_curve),
     METH_VARARGS | METH_KEYWORDS, "Compute a rectilinear finite-source light curve from observation times."},
    {
        "fspl_magnification",
        reinterpret_cast<PyCFunction>(finite_source_magnification),
        METH_VARARGS | METH_KEYWORDS,
        "Evaluate finite-source point-lens magnification in the compiled kernel."
    },
    {
        "fspl_magnification_gradient",
        reinterpret_cast<PyCFunction>(finite_source_magnification_gradient),
        METH_VARARGS | METH_KEYWORDS,
        "Finite-source magnification with analytic derivatives (A, dA/du, dA/drho)."
    },
    {
        "score_fspl_seeds",
        reinterpret_cast<PyCFunction>(score_finite_source_seeds),
        METH_VARARGS | METH_KEYWORDS,
        "Score a batch of FSPL seeds with profiled source and blend fluxes."
    },
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_vbm_cpp",
    "Compiled finite-source magnification backend for singlelensfitter.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__vbm_cpp(void) {
    import_array();
    return PyModule_Create(&module);
}
