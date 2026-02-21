find_package(GTest CONFIG REQUIRED)

add_executable(sense8_common_tests
  common_version_test.cpp
  euroc_dataset_test.cpp
  replay_cursor_test.cpp
)

target_link_libraries(sense8_common_tests
  PRIVATE
    GTest::gtest_main
    sense8_common
    sense8_io
)

target_compile_definitions(sense8_common_tests
  PRIVATE
    SENSE8_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

include(GoogleTest)
gtest_discover_tests(sense8_common_tests)
