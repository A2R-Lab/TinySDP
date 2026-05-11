cmake_minimum_required(VERSION 3.15)

project(TinySDP VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Dense
        PATHS /opt/homebrew/include/eigen3 /usr/local/include/eigen3 /usr/include/eigen3
    )
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 not found. Install Eigen or set EIGEN3_INCLUDE_DIR.")
    endif()
endif()

add_library(solver STATIC
    src/solver/admm.cpp
    src/solver/tiny_api.cpp
)

target_include_directories(solver
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(Eigen3_FOUND)
    target_link_libraries(solver PUBLIC Eigen3::Eigen)
else()
    target_include_directories(solver PUBLIC ${EIGEN3_INCLUDE_DIR})
endif()

target_compile_features(solver PUBLIC cxx_std_17)

add_subdirectory(examples)
