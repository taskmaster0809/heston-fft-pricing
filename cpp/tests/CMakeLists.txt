include(CTest)
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)

add_executable(heston_tests
    test_mc.cpp
    test_fft.cpp
    ../src/heston_mc.cpp
    ../src/heston_fft.cpp
    ../src/utils.cpp
)

target_include_directories(heston_tests PRIVATE ../include ${FFTW_INCLUDE_DIRS})
target_link_libraries(heston_tests PRIVATE 
    Catch2::Catch2WithMain
    PkgConfig::FFTW
)

catch_discover_tests(heston_tests)