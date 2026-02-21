find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(sense8_viewer
  main.cpp
)

target_link_libraries(sense8_viewer
  PRIVATE
    imgui::imgui
    glfw
    OpenGL::GL
    sense8_io
    sense8_common
)

target_compile_features(sense8_viewer PRIVATE cxx_std_20)
