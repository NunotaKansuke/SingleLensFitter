SingleLensFitter originated in jacscanomaly and retains its MIT copyright notice.

The native extensions compile the unmodified VBMicrolensing library supplied
by the build dependency. VBMicrolensing is by Valerio Bozza and collaborators
and is distributed under LGPL version 3. To rebuild against modified library
sources, set `VBMICROLENSING_SOURCE_DIR` to the directory containing
`VBMicrolensingLibrary.cpp` and its header before building the package.
See the [upstream source and citation instructions](https://github.com/valboz/VBMicrolensing)
and the license texts in `licenses/`.

The bundled Earth ephemeris is from NASA/JPL Horizons. Query and reference-frame
details are preserved in the header of the original table.
