#include <pybind11/pybind11.h>
#include "heston_mc.h"
#include "heston_fft.h"
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(heston, m){
    m.def("heston_mc_price", &heston_mc_price, "Price a European call under Heston via Monte Carlo",
    py::arg("S0"), 
    py::arg("K"), 
    py::arg("v0"), 
    py::arg("r"), 
    py::arg("rho"),
    py::arg("kappa"), 
    py::arg("theta"), 
    py::arg("xi"), 
    py::arg("T"),
    py::arg("N")=252, 
    py::arg("num_paths")=1000
    );

    m.def("black_scholes_price", &black_scholes_price, "Price a European call under Black-Scholes via analytical solution",
    py::arg("S"),
    py::arg("K"),
    py::arg("T"),
    py::arg("r"),
    py::arg("sigma")
    );
}
