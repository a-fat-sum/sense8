#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include <CLI/CLI.hpp>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "sense8/io/euroc_dataset.hpp"
#include "sense8/io/replay_cursor.hpp"
#include "sense8/io/sensor_packets.hpp"

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

namespace {

struct FrameTexture {
  GLuint texture_id = 0;
  int width = 0;
  int height = 0;
  std::string loaded_image_path;
  std::string failed_image_path;
};

void EnsureTexture(FrameTexture* frame_texture) {
  if (frame_texture->texture_id != 0) {
    return;
  }

  glGenTextures(1, &frame_texture->texture_id);
  glBindTexture(GL_TEXTURE_2D, frame_texture->texture_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void DestroyTexture(FrameTexture* frame_texture) {
  if (frame_texture->texture_id != 0) {
    glDeleteTextures(1, &frame_texture->texture_id);
    frame_texture->texture_id = 0;
  }
  frame_texture->width = 0;
  frame_texture->height = 0;
  frame_texture->loaded_image_path.clear();
  frame_texture->failed_image_path.clear();
}

std::size_t FindCameraIndexForTimestamp(const std::vector<sense8::io::CameraFrame>& camera_frames, std::int64_t timestamp_ns) {
  if (camera_frames.empty()) {
    return camera_frames.size();
  }

  const auto it = std::upper_bound(
      camera_frames.begin(), camera_frames.end(), timestamp_ns, [](std::int64_t ts, const sense8::io::CameraFrame& frame) {
        return ts < frame.timestamp_ns;
      });

  if (it == camera_frames.begin()) {
    return camera_frames.size();
  }

  return static_cast<std::size_t>(std::distance(camera_frames.begin(), it - 1));
}

bool UpdateFrameTexture(const std::vector<sense8::io::CameraFrame>& camera_frames,
                        std::int64_t timestamp_ns,
                        FrameTexture* frame_texture,
                        std::string* error_message) {
  const std::size_t camera_index = FindCameraIndexForTimestamp(camera_frames, timestamp_ns);
  if (camera_index >= camera_frames.size()) {
    if (error_message != nullptr) {
      *error_message = "No camera frame available at this timestamp yet";
    }
    return false;
  }

  const std::string& image_path = camera_frames[camera_index].image_path;
  if (frame_texture->loaded_image_path == image_path && frame_texture->texture_id != 0) {
    return true;
  }

  if (frame_texture->failed_image_path == image_path) {
    if (error_message != nullptr) {
      *error_message = std::string("Failed to read image: ") + image_path;
    }
    return false;
  }

  const cv::Mat bgr_image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr_image.empty()) {
    frame_texture->failed_image_path = image_path;
    if (error_message != nullptr) {
      *error_message = std::string("Failed to read image: ") + image_path;
    }
    return false;
  }

  cv::Mat rgba_image;
  cv::cvtColor(bgr_image, rgba_image, cv::COLOR_BGR2RGBA);

  EnsureTexture(frame_texture);
  glBindTexture(GL_TEXTURE_2D, frame_texture->texture_id);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(
      GL_TEXTURE_2D,
      0,
      GL_RGBA,
      rgba_image.cols,
      rgba_image.rows,
      0,
      GL_RGBA,
      GL_UNSIGNED_BYTE,
      rgba_image.data);
  glBindTexture(GL_TEXTURE_2D, 0);

  frame_texture->width = rgba_image.cols;
  frame_texture->height = rgba_image.rows;
  frame_texture->loaded_image_path = image_path;
  frame_texture->failed_image_path.clear();
  if (error_message != nullptr) {
    error_message->clear();
  }

  return true;
}

std::size_t FindImuUpperBound(const std::vector<sense8::io::ImuSample>& imu_samples, std::int64_t timestamp_ns) {
  const auto it = std::upper_bound(
      imu_samples.begin(), imu_samples.end(), timestamp_ns, [](std::int64_t ts, const sense8::io::ImuSample& sample) {
        return ts < sample.timestamp_ns;
      });
  return static_cast<std::size_t>(std::distance(imu_samples.begin(), it));
}

std::size_t FindImuLowerBound(const std::vector<sense8::io::ImuSample>& imu_samples, std::int64_t timestamp_ns) {
  const auto it = std::lower_bound(
      imu_samples.begin(), imu_samples.end(), timestamp_ns, [](const sense8::io::ImuSample& sample, std::int64_t ts) {
        return sample.timestamp_ns < ts;
      });
  return static_cast<std::size_t>(std::distance(imu_samples.begin(), it));
}

std::vector<float> CollectImuAxisValues(const std::vector<sense8::io::ImuSample>& imu_samples,
                                        std::size_t begin_index,
                                        std::size_t end_index,
                                        bool gyro_axis,
                                        int axis_index) {
  std::vector<float> values;
  values.reserve(end_index - begin_index);
  for (std::size_t index = begin_index; index < end_index; ++index) {
    const auto& sample = imu_samples[index];
    const double value = gyro_axis ? sample.angular_velocity_rad_s[axis_index] : sample.linear_acceleration_m_s2[axis_index];
    values.push_back(static_cast<float>(value));
  }
  return values;
}

void DrawAxisPlot(const char* label, const std::vector<float>& values, float min_v, float max_v) {
  if (values.empty()) {
    ImGui::Text("%s: no samples in current window", label);
    return;
  }

  ImGui::PlotLines(label, values.data(), static_cast<int>(values.size()), 0, nullptr, min_v, max_v, ImVec2(0.0F, 70.0F));
}

void DrawImuPlots(const std::vector<sense8::io::ImuSample>& imu_samples, std::int64_t current_timestamp_ns, float window_seconds) {
  if (imu_samples.empty()) {
    ImGui::Text("No IMU samples available");
    return;
  }

  const auto window_ns = static_cast<std::int64_t>(window_seconds * 1e9F);
  const auto begin_ts = current_timestamp_ns > window_ns ? current_timestamp_ns - window_ns : imu_samples.front().timestamp_ns;
  const std::size_t begin_index = FindImuLowerBound(imu_samples, begin_ts);
  const std::size_t end_index = FindImuUpperBound(imu_samples, current_timestamp_ns);

  if (begin_index >= end_index) {
    ImGui::Text("No IMU samples available for this time cursor");
    return;
  }

  ImGui::Text("Window: last %.2f s (%zu samples)", window_seconds, end_index - begin_index);

  const std::array<const char*, 3> axis_names = {"X", "Y", "Z"};

  ImGui::SeparatorText("Gyro (rad/s)");
  for (int axis = 0; axis < 3; ++axis) {
    const std::vector<float> values = CollectImuAxisValues(imu_samples, begin_index, end_index, true, axis);
    const std::string plot_label = std::string("gyro ") + axis_names[axis];
    DrawAxisPlot(plot_label.c_str(), values, -4.0F, 4.0F);
  }

  ImGui::SeparatorText("Accel (m/s^2)");
  for (int axis = 0; axis < 3; ++axis) {
    const std::vector<float> values = CollectImuAxisValues(imu_samples, begin_index, end_index, false, axis);
    const std::string plot_label = std::string("accel ") + axis_names[axis];
    DrawAxisPlot(plot_label.c_str(), values, -20.0F, 20.0F);
  }
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"sense8 replay viewer"};
  std::string dataset_path;
  app.add_option("--dataset", dataset_path, "Path to EuRoC dataset root (contains mav0)")->required();

  CLI11_PARSE(app, argc, argv);

  sense8::io::SensorSequence sequence;
  sense8::io::ReplayCursor cursor;
  try {
    const sense8::io::EurocDatasetLoader loader;
    sequence = loader.Load(dataset_path);
    cursor.Bind(&sequence.merged_packets);
  } catch (const std::exception& ex) {
    spdlog::error("Failed to load dataset: {}", ex.what());
    return EXIT_FAILURE;
  }

  if (!glfwInit()) {
    return EXIT_FAILURE;
  }

  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  GLFWwindow* window = glfwCreateWindow(1280, 720, "sense8 viewer", nullptr, nullptr);
  if (window == nullptr) {
    glfwTerminate();
    return EXIT_FAILURE;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  double previous_time_s = glfwGetTime();
  float timeline_ratio = 0.0F;
  float imu_window_seconds = 5.0F;
  FrameTexture frame_texture;
  std::string frame_error;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    const double current_time_s = glfwGetTime();
    const double delta_time_s = current_time_s - previous_time_s;
    previous_time_s = current_time_s;
    cursor.Update(delta_time_s);
    timeline_ratio = static_cast<float>(cursor.ratio());

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("sense8 replay");
    if (ImGui::BeginTabBar("top_tabs")) {
      if (ImGui::BeginTabItem("Replay")) {
        if (ImGui::Button(cursor.playing() ? "Pause" : "Play")) {
          cursor.TogglePlaying();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
          cursor.Reset();
          timeline_ratio = 0.0F;
        }

        float playback_speed = static_cast<float>(cursor.speed());
        if (ImGui::SliderFloat("Speed", &playback_speed, 0.1F, 4.0F, "%.1fx")) {
          cursor.SetSpeed(playback_speed);
        }

        if (ImGui::SliderFloat("Timeline", &timeline_ratio, 0.0F, 1.0F, "%.3f")) {
          cursor.SeekToRatio(timeline_ratio);
        }

        ImGui::Separator();
        ImGui::Text("Dataset: %s", dataset_path.c_str());
        ImGui::Text("IMU samples: %zu", sequence.imu_samples.size());
        ImGui::Text("Camera frames: %zu", sequence.camera_frames.size());
        ImGui::Text("Merged packets: %zu", sequence.merged_packets.size());
        ImGui::Text("Current packet index: %zu", cursor.packet_index());
        ImGui::Text("Timestamp: %lld ns", static_cast<long long>(cursor.current_timestamp_ns()));

        const bool has_frame_texture = UpdateFrameTexture(
            sequence.camera_frames,
            cursor.current_timestamp_ns(),
            &frame_texture,
            &frame_error);

        ImGui::SeparatorText("Camera View");
        if (has_frame_texture) {
          const float available_width = ImGui::GetContentRegionAvail().x;
          const float aspect_ratio = static_cast<float>(frame_texture.height) / static_cast<float>(frame_texture.width);
          const float image_width = std::max(320.0F, available_width);
          const float image_height = image_width * aspect_ratio;
          ImGui::Image((ImTextureID)(intptr_t)frame_texture.texture_id, ImVec2(image_width, image_height));
          ImGui::TextWrapped("Image: %s", frame_texture.loaded_image_path.c_str());
        } else {
          ImGui::TextWrapped("%s", frame_error.c_str());
        }

        if (!sequence.merged_packets.empty()) {
          const auto& packet = sequence.merged_packets[cursor.packet_index()];
          if (std::holds_alternative<sense8::io::ImuSample>(packet.payload)) {
            ImGui::Text("Payload: IMU");
          } else {
            ImGui::Text("Payload: Camera");
          }
        }

        ImGui::EndTabItem();
      }

      if (ImGui::BeginTabItem("IMU Plots")) {
        ImGui::SliderFloat("Window (s)", &imu_window_seconds, 0.5F, 30.0F, "%.1f s");
        DrawImuPlots(sequence.imu_samples, cursor.current_timestamp_ns(), imu_window_seconds);
        ImGui::EndTabItem();
      }

      ImGui::EndTabBar();
    }

    ImGui::End();

    ImGui::Render();
    int display_w = 0;
    int display_h = 0;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.08f, 0.08f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  DestroyTexture(&frame_texture);

  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
