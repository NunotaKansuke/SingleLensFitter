from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


VBM_CPP_SOURCE = Path("src/singlelensfitter/_vbm_cpp.cpp")
PARALLAX_CPP_SOURCE = Path("src/singlelensfitter/_parallax_cpp.cpp")


def _find_vbm_sources() -> tuple[Path, Path] | None:
    """Locate the C++ sources installed by the VBMicrolensing build dependency.

    Native FSPL/parallax support is part of the standard build.
    ``VBMICROLENSING_SOURCE_DIR`` supports non-standard installations.
    """
    candidates: list[Path] = []
    source_dir = os.environ.get("VBMICROLENSING_SOURCE_DIR")
    if source_dir:
        candidates.append(Path(source_dir))
    try:
        import VBMicrolensing  # type: ignore

        candidates.append(Path(VBMicrolensing.__file__).resolve().parent / "lib")
    except ImportError:
        pass
    for directory in candidates:
        if (directory / "VBMicrolensingLibrary.cpp").is_file() and (directory / "VBMicrolensingLibrary.h").is_file():
            return directory / "VBMicrolensingLibrary.cpp", directory
    return None


def _compile_and_link_args() -> tuple[list[str], list[str]]:
    """Return platform-specific flags for the mandatory C++ backend."""
    if sys.platform == "win32":
        return ["/O2", "/std:c++17", "/openmp"], []
    if sys.platform == "darwin":
        # Apple does not ship an OpenMP runtime.  macOS wheels disable the
        # OpenMP pragmas so they do not depend on the Homebrew libomp version
        # or on a particular macOS deployment target.  Local source builds
        # keep OpenMP enabled unless this switch is set explicitly.
        if os.environ.get("SINGLELENSFITTER_DISABLE_OPENMP", "").lower() in {"1", "true", "yes"}:
            return ["-O3", "-fno-math-errno", "-std=c++17"], []
        return ["-O3", "-fno-math-errno", "-std=c++17", "-Xpreprocessor", "-fopenmp"], ["-lomp"]
    return ["-O3", "-fno-math-errno", "-std=c++17", "-fopenmp"], ["-fopenmp"]


class build_ext(_build_ext):
    """Build the C++ backend as a required extension.

    The default workflow uses the compiled anomaly grid and the canonical
    SciPy-based fitter API, so the package should fail during install if the
    compiled backend cannot be built instead of installing a package that
    later fails at runtime.
    """

    def build_extensions(self) -> None:
        vbm_sources = _find_vbm_sources()
        if vbm_sources is None:
            raise RuntimeError(
                "VBMicrolensing C++ sources are required to build singlelensfitter's "
                "native FSPL/parallax backend. Install VBMicrolensing>=5.5 or set "
                "VBMICROLENSING_SOURCE_DIR."
            )
        _, vbm_include = vbm_sources
        compile_args, link_args = _compile_and_link_args()
        for ext in self.extensions:
            if ext.name in {"singlelensfitter._vbm_cpp", "singlelensfitter._parallax_cpp"}:
                ext.include_dirs = list(ext.include_dirs or []) + [str(vbm_include)]
            if ext.name == "singlelensfitter._vbm_cpp" and os.environ.get("SINGLELENSFITTER_XLA_INCLUDE"):
                ext.include_dirs.append(os.environ["SINGLELENSFITTER_XLA_INCLUDE"])
                ext.define_macros.append(("SINGLELENSFITTER_XLA_FFI", "1"))
            ext.extra_compile_args = list(ext.extra_compile_args or []) + compile_args
            ext.extra_link_args = list(ext.extra_link_args or []) + link_args

        try:
            super().build_extensions()
        except Exception as exc:
            raise RuntimeError(
                "Failed to build the singlelensfitter C++ extensions. "
                "Make sure a C++17 compiler and OpenMP runtime are available."
            ) from exc


magnification_headers = [
    "src/singlelensfitter/cpp/chebyshev_patch.hpp",
    "src/singlelensfitter/cpp/uniform_limb_table.hpp",
    "src/singlelensfitter/cpp/uniform_table_data.hpp",
    "src/singlelensfitter/cpp/mu_quadrature.hpp",
    "src/singlelensfitter/cpp/mu_table_data.hpp",
    "src/singlelensfitter/cpp/linear_limb_darkening.hpp",
    "src/singlelensfitter/cpp/source_limb.hpp",
    "src/singlelensfitter/cpp/vbm_magnification.hpp",
]
extensions = []

extensions.append(
    Extension(
        "singlelensfitter._vbm_cpp",
        [str(VBM_CPP_SOURCE), "src/singlelensfitter/vbm_library.cpp"],
        include_dirs=[np.get_include(), str(VBM_CPP_SOURCE.parent)],
        language="c++",
        depends=[str(VBM_CPP_SOURCE), *magnification_headers],
    )
)
extensions.append(
    Extension(
        "singlelensfitter._parallax_cpp",
        [str(PARALLAX_CPP_SOURCE), "src/singlelensfitter/vbm_library.cpp"],
        include_dirs=[np.get_include(), str(PARALLAX_CPP_SOURCE.parent)],
        language="c++",
        depends=[
            str(PARALLAX_CPP_SOURCE),
            "src/singlelensfitter/cpp/ephemeris.hpp",
            "src/singlelensfitter/cpp/sky_projection.hpp",
            "src/singlelensfitter/cpp/parallax_trajectory.hpp",
            *magnification_headers,
        ],
    )
)

setup(
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext},
)
