cmake_minimum_required(VERSION 3.15)

project(HestonModel)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.4
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(pybind11)

add_subdirectory(cpp)