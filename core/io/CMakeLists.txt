add_library(sense8_io STATIC
  src/euroc_dataset.cpp
  src/replay_cursor.cpp
)

target_include_directories(sense8_io
  PUBLIC
    include
)

target_link_libraries(sense8_io
  PUBLIC
    fmt::fmt
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    sense8_common
)

target_compile_features(sense8_io PUBLIC cxx_std_20)
