cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(libtorch_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3")

add_executable(libtorch_benchmark main.cpp)
target_link_libraries(libtorch_benchmark "${TORCH_LIBRARIES}")

set_target_properties(libtorch_benchmark PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
