add_executable(sense8_vio_main
  main.cpp
)

target_include_directories(sense8_vio_main
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(sense8_vio_main
  PRIVATE
    CLI11::CLI11
    ${OpenCV_LIBS}
    sense8_io
    sense8_common
)

target_compile_features(sense8_vio_main PRIVATE cxx_std_20)
