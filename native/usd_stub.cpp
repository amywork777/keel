// Stand-in for SmPyUsd.cpp (which needs OpenUSD): an empty `usd` submodule, so the rest
// of NVIDIA's bindings build and import unchanged.
#include "SmPyCommon.h"

namespace py = pybind11;

void bind_usd(py::module_& parent)
{
    py::module_ usd = parent.def_submodule(
        "usd", "Not available in this native build (built without OpenUSD).");
    usd.attr("AVAILABLE") = false;
}
