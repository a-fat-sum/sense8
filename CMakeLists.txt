cmake_minimum_required(VERSION 3.24)

project(sense8 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SENSE8_BUILD_TESTS "Build unit tests" ON)
option(SENSE8_BUILD_VIEWER "Build Dear ImGui viewer app" OFF)

find_package(CLI11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc calib3d features2d video)
find_package(spdlog CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_subdirectory(core/common)
add_subdirectory(core/io)
add_subdirectory(apps/sense8_vio_main)

if(SENSE8_BUILD_VIEWER)
  add_subdirectory(apps/sense8_viewer)
endif()

if(SENSE8_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
