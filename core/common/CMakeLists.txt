add_library(sense8_common STATIC
  src/version.cpp
)

target_include_directories(sense8_common
  PUBLIC
    include
)

target_link_libraries(sense8_common
  PUBLIC
    Eigen3::Eigen
    fmt::fmt
    spdlog::spdlog
    yaml-cpp::yaml-cpp
)

target_compile_features(sense8_common PUBLIC cxx_std_20)
