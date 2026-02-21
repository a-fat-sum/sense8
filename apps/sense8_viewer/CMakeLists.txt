find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(sense8_viewer
  main.cpp
)

target_include_directories(sense8_viewer
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(sense8_viewer
  PRIVATE
    CLI11::CLI11
    imgui::imgui
    glfw
    OpenGL::GL
    ${OpenCV_LIBS}
    sense8_io
    sense8_common
)

target_compile_features(sense8_viewer PRIVATE cxx_std_20)
