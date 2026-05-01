pybind11_add_module(heston 
bindings/bindings.cpp
src/heston_mc.cpp
src/heston_fft.cpp
src/utils.cpp
)

target_include_directories(heston PRIVATE include)