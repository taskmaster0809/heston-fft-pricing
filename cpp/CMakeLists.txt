find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3 IMPORTED_TARGET)

pybind11_add_module(heston 
bindings/bindings.cpp
src/heston_mc.cpp
src/heston_fft.cpp
src/utils.cpp
)

target_include_directories(heston PRIVATE include ${FFTW_INCLUDE_DIRS})
target_link_libraries(heston PRIVATE PkgConfig::FFTW)

add_subdirectory(tests)