find_package(GTest CONFIG REQUIRED)

add_executable(sense8_common_tests
  common_version_test.cpp
)

target_link_libraries(sense8_common_tests
  PRIVATE
    GTest::gtest_main
    sense8_common
)

include(GoogleTest)
gtest_discover_tests(sense8_common_tests)
