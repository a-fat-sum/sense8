#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <CLI/CLI.hpp>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

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
  std::vector<cv::KeyPoint> orb_keypoints;
  double orb_compute_ms = 0.0;
  bool orb_enabled_for_texture = false;
  int orb_max_features_for_texture = 0;
  bool tracking_enabled_for_texture = false;
};

struct FrontendTrack {
  cv::Point2f previous_point;
  cv::Point2f current_point;
};

struct FrontendMetrics {
  int keypoints_current = 0;
  int mutual_matches = 0;
  int inlier_matches = 0;
  double detect_compute_ms = 0.0;
  double match_ms = 0.0;
  double ransac_ms = 0.0;
  std::string geometric_model = "none";
};

struct FrontendTrackingState {
  std::vector<cv::KeyPoint> previous_keypoints;
  cv::Mat previous_descriptors;
  std::size_t previous_camera_index = 0;
  std::string previous_image_path;
  bool has_previous_frame = false;

  std::vector<FrontendTrack> inlier_tracks;
  FrontendMetrics metrics;
  std::string match_previous_image_path;
  std::string match_current_image_path;
  std::vector<cv::KeyPoint> match_previous_keypoints;
  std::vector<cv::KeyPoint> match_current_keypoints;
  bool has_match_pair = false;

  cv::Mat camera_matrix;
  bool has_camera_intrinsics = false;
};

bool LoadEurocIntrinsics(const std::filesystem::path& dataset_root, cv::Mat* camera_matrix, std::string* error_message) {
  try {
    const auto sensor_yaml_path = dataset_root / "mav0" / "cam0" / "sensor.yaml";
    if (!std::filesystem::exists(sensor_yaml_path)) {
      if (error_message != nullptr) {
        *error_message = "cam0/sensor.yaml not found; using fallback geometric model";
      }
      return false;
    }

    const YAML::Node root = YAML::LoadFile(sensor_yaml_path.string());
    const YAML::Node intrinsics = root["intrinsics"];
    if (!intrinsics || !intrinsics.IsSequence() || intrinsics.size() < 4) {
      if (error_message != nullptr) {
        *error_message = "Invalid intrinsics array in cam0/sensor.yaml";
      }
      return false;
    }

    const double fx = intrinsics[0].as<double>();
    const double fy = intrinsics[1].as<double>();
    const double cx = intrinsics[2].as<double>();
    const double cy = intrinsics[3].as<double>();

    *camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    return true;
  } catch (const std::exception& ex) {
    if (error_message != nullptr) {
      *error_message = std::string("Failed to load cam0 intrinsics: ") + ex.what();
    }
    return false;
  }
}

std::vector<cv::DMatch> ComputeMutualRatioMatches(const cv::Mat& previous_descriptors, const cv::Mat& current_descriptors) {
  constexpr float kRatioThreshold = 0.75F;

  cv::BFMatcher matcher(cv::NORM_HAMMING, false);
  std::vector<std::vector<cv::DMatch>> forward_knn;
  std::vector<std::vector<cv::DMatch>> reverse_knn;
  matcher.knnMatch(previous_descriptors, current_descriptors, forward_knn, 2);
  matcher.knnMatch(current_descriptors, previous_descriptors, reverse_knn, 2);

  std::vector<cv::DMatch> forward_ratio;
  forward_ratio.reserve(forward_knn.size());
  for (const auto& candidate_set : forward_knn) {
    if (candidate_set.size() < 2) {
      continue;
    }
    if (candidate_set[0].distance < kRatioThreshold * candidate_set[1].distance) {
      forward_ratio.push_back(candidate_set[0]);
    }
  }

  std::set<std::pair<int, int>> reverse_ratio_pairs;
  for (const auto& candidate_set : reverse_knn) {
    if (candidate_set.size() < 2) {
      continue;
    }
    if (candidate_set[0].distance < kRatioThreshold * candidate_set[1].distance) {
      reverse_ratio_pairs.insert({candidate_set[0].queryIdx, candidate_set[0].trainIdx});
    }
  }

  std::vector<cv::DMatch> mutual_matches;
  mutual_matches.reserve(forward_ratio.size());
  for (const auto& match : forward_ratio) {
    const auto reverse_key = std::pair<int, int>{match.trainIdx, match.queryIdx};
    if (reverse_ratio_pairs.contains(reverse_key)) {
      mutual_matches.push_back(match);
    }
  }

  return mutual_matches;
}

void ResetTrackingState(FrontendTrackingState* tracking_state) {
  tracking_state->previous_keypoints.clear();
  tracking_state->previous_descriptors.release();
  tracking_state->previous_camera_index = 0;
  tracking_state->previous_image_path.clear();
  tracking_state->has_previous_frame = false;
  tracking_state->inlier_tracks.clear();
  tracking_state->metrics = FrontendMetrics{};
  tracking_state->match_previous_image_path.clear();
  tracking_state->match_current_image_path.clear();
  tracking_state->match_previous_keypoints.clear();
  tracking_state->match_current_keypoints.clear();
  tracking_state->has_match_pair = false;
}

void UpdateTracking(const std::vector<cv::KeyPoint>& current_keypoints,
                    const cv::Mat& current_descriptors,
                    std::size_t camera_index,
                    const std::string& current_image_path,
                    FrontendTrackingState* tracking_state) {
  tracking_state->inlier_tracks.clear();
  tracking_state->metrics = FrontendMetrics{};
  tracking_state->metrics.keypoints_current = static_cast<int>(current_keypoints.size());
  tracking_state->has_match_pair = false;
  tracking_state->match_previous_image_path.clear();
  tracking_state->match_current_image_path.clear();
  tracking_state->match_previous_keypoints.clear();
  tracking_state->match_current_keypoints.clear();

  if (current_descriptors.empty() || current_keypoints.empty()) {
    tracking_state->previous_keypoints = current_keypoints;
    tracking_state->previous_descriptors = current_descriptors.clone();
    tracking_state->previous_camera_index = camera_index;
    tracking_state->previous_image_path = current_image_path;
    tracking_state->has_previous_frame = true;
    return;
  }

  if (!tracking_state->has_previous_frame ||
      tracking_state->previous_descriptors.empty() ||
      camera_index != tracking_state->previous_camera_index + 1) {
    tracking_state->previous_keypoints = current_keypoints;
    tracking_state->previous_descriptors = current_descriptors.clone();
    tracking_state->previous_camera_index = camera_index;
    tracking_state->previous_image_path = current_image_path;
    tracking_state->has_previous_frame = true;
    return;
  }

  tracking_state->match_previous_image_path = tracking_state->previous_image_path;
  tracking_state->match_current_image_path = current_image_path;
  tracking_state->match_previous_keypoints = tracking_state->previous_keypoints;
  tracking_state->match_current_keypoints = current_keypoints;
  tracking_state->has_match_pair = true;

  const auto match_start = std::chrono::steady_clock::now();
  const std::vector<cv::DMatch> mutual_matches = ComputeMutualRatioMatches(
      tracking_state->previous_descriptors,
      current_descriptors);
  const auto match_end = std::chrono::steady_clock::now();

  tracking_state->metrics.mutual_matches = static_cast<int>(mutual_matches.size());
  tracking_state->metrics.match_ms = std::chrono::duration<double, std::milli>(match_end - match_start).count();

  if (mutual_matches.size() >= 8) {
    std::vector<cv::Point2f> previous_points;
    std::vector<cv::Point2f> current_points;
    previous_points.reserve(mutual_matches.size());
    current_points.reserve(mutual_matches.size());

    for (const auto& match : mutual_matches) {
      previous_points.push_back(tracking_state->previous_keypoints[match.queryIdx].pt);
      current_points.push_back(current_keypoints[match.trainIdx].pt);
    }

    const auto ransac_start = std::chrono::steady_clock::now();
    std::vector<unsigned char> inlier_mask;

    if (tracking_state->has_camera_intrinsics && !tracking_state->camera_matrix.empty()) {
      tracking_state->metrics.geometric_model = "Essential";
      cv::findEssentialMat(previous_points, current_points, tracking_state->camera_matrix, cv::RANSAC, 0.999, 1.0, inlier_mask);
    } else {
      tracking_state->metrics.geometric_model = "Fundamental";
      cv::findFundamentalMat(previous_points, current_points, cv::FM_RANSAC, 1.0, 0.999, inlier_mask);
    }

    const auto ransac_end = std::chrono::steady_clock::now();
    tracking_state->metrics.ransac_ms = std::chrono::duration<double, std::milli>(ransac_end - ransac_start).count();

    for (std::size_t index = 0; index < inlier_mask.size(); ++index) {
      if (inlier_mask[index] == 0) {
        continue;
      }
      tracking_state->inlier_tracks.push_back(FrontendTrack{previous_points[index], current_points[index]});
    }
  }

  tracking_state->metrics.inlier_matches = static_cast<int>(tracking_state->inlier_tracks.size());

  tracking_state->previous_keypoints = current_keypoints;
  tracking_state->previous_descriptors = current_descriptors.clone();
  tracking_state->previous_camera_index = camera_index;
  tracking_state->previous_image_path = current_image_path;
  tracking_state->has_previous_frame = true;
}

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
  frame_texture->orb_keypoints.clear();
  frame_texture->orb_compute_ms = 0.0;
  frame_texture->orb_enabled_for_texture = false;
  frame_texture->orb_max_features_for_texture = 0;
}

bool UpdateTextureFromImagePath(const std::string& image_path,
                                FrameTexture* frame_texture,
                                std::string* error_message) {
  if (image_path.empty()) {
    if (error_message != nullptr) {
      *error_message = "Image path is empty";
    }
    return false;
  }

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

void ComputeOrbFeatures(const cv::Mat& bgr_image,
                        int max_features,
                        std::vector<cv::KeyPoint>* keypoints,
                        double* compute_ms) {
  const auto start = std::chrono::steady_clock::now();
  const auto detector = cv::ORB::create(max_features);
  detector->detect(bgr_image, *keypoints);
  const auto end = std::chrono::steady_clock::now();
  *compute_ms = std::chrono::duration<double, std::milli>(end - start).count();
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
                        bool compute_orb,
                        int orb_max_features,
                        bool compute_tracking,
                        FrontendTrackingState* tracking_state,
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
  const bool orb_settings_match = frame_texture->orb_enabled_for_texture == compute_orb &&
                                  frame_texture->orb_max_features_for_texture == orb_max_features;
  const bool tracking_settings_match = frame_texture->tracking_enabled_for_texture == compute_tracking;
  if (frame_texture->loaded_image_path == image_path && frame_texture->texture_id != 0 && orb_settings_match && tracking_settings_match) {
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

  std::vector<cv::KeyPoint> detected_keypoints;
  cv::Mat detected_descriptors;
  frame_texture->orb_compute_ms = 0.0;
  if (compute_orb || compute_tracking) {
    const auto detect_start = std::chrono::steady_clock::now();
    const auto detector = cv::ORB::create(orb_max_features);
    detector->detectAndCompute(bgr_image, cv::Mat(), detected_keypoints, detected_descriptors);
    const auto detect_end = std::chrono::steady_clock::now();
    frame_texture->orb_compute_ms = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
    frame_texture->orb_keypoints = detected_keypoints;
  } else {
    frame_texture->orb_keypoints.clear();
  }

  if (compute_tracking && tracking_state != nullptr) {
    UpdateTracking(detected_keypoints, detected_descriptors, camera_index, image_path, tracking_state);
    tracking_state->metrics.detect_compute_ms = frame_texture->orb_compute_ms;
  } else if (tracking_state != nullptr) {
    ResetTrackingState(tracking_state);
  }

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
  frame_texture->orb_enabled_for_texture = compute_orb;
  frame_texture->orb_max_features_for_texture = orb_max_features;
  frame_texture->tracking_enabled_for_texture = compute_tracking;
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

std::vector<cv::Point3f> LoadEurocGroundTruthPositions(const std::filesystem::path& dataset_root) {
  std::vector<cv::Point3f> positions;
  const auto gt_csv = dataset_root / "mav0" / "state_groundtruth_estimate0" / "data.csv";
  std::ifstream file(gt_csv);
  if (!file.is_open()) {
    return positions;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream ss(line);
    std::array<std::string, 4> fields;
    bool valid = true;
    for (std::size_t index = 0; index < fields.size(); ++index) {
      if (!std::getline(ss, fields[index], ',')) {
        valid = false;
        break;
      }
    }

    if (!valid) {
      continue;
    }

    try {
      const float px = std::stof(fields[1]);
      const float py = std::stof(fields[2]);
      const float pz = std::stof(fields[3]);
      positions.emplace_back(px, py, pz);
    } catch (const std::exception&) {
      continue;
    }
  }

  if (!positions.empty()) {
    const cv::Point3f origin = positions.front();
    for (auto& point : positions) {
      point -= origin;
    }
  }

  return positions;
}

std::vector<cv::Point3f> TriangulateDebugPoints(const FrontendTrackingState& tracking_state,
                                                const std::vector<FrontendTrack>& tracks,
                                                std::size_t max_points) {
  std::vector<cv::Point3f> triangulated_points;
  if (!tracking_state.has_camera_intrinsics || tracking_state.camera_matrix.empty() || tracks.size() < 8) {
    return triangulated_points;
  }

  const std::size_t track_count = std::min(tracks.size(), max_points);
  std::vector<cv::Point2f> previous_points;
  std::vector<cv::Point2f> current_points;
  previous_points.reserve(track_count);
  current_points.reserve(track_count);
  for (std::size_t index = 0; index < track_count; ++index) {
    previous_points.push_back(tracks[index].previous_point);
    current_points.push_back(tracks[index].current_point);
  }

  cv::Mat inlier_mask;
  const cv::Mat essential = cv::findEssentialMat(
      previous_points,
      current_points,
      tracking_state.camera_matrix,
      cv::RANSAC,
      0.999,
      1.0,
      inlier_mask);
  if (essential.empty()) {
    return triangulated_points;
  }

  cv::Mat rotation;
  cv::Mat translation;
  const int recovered = cv::recoverPose(
      essential,
      previous_points,
      current_points,
      tracking_state.camera_matrix,
      rotation,
      translation,
      inlier_mask);
  if (recovered < 8) {
    return triangulated_points;
  }

  const cv::Mat projection_1 = tracking_state.camera_matrix * cv::Mat::eye(3, 4, CV_64F);
  cv::Mat pose_2;
  cv::hconcat(rotation, translation, pose_2);
  const cv::Mat projection_2 = tracking_state.camera_matrix * pose_2;

  cv::Mat homogeneous_points;
  cv::triangulatePoints(projection_1, projection_2, previous_points, current_points, homogeneous_points);

  triangulated_points.reserve(static_cast<std::size_t>(homogeneous_points.cols));
  for (int column = 0; column < homogeneous_points.cols; ++column) {
    if (!inlier_mask.empty() && inlier_mask.at<unsigned char>(column) == 0) {
      continue;
    }

    const double w = homogeneous_points.at<double>(3, column);
    if (std::abs(w) < 1e-9) {
      continue;
    }

    const double x = homogeneous_points.at<double>(0, column) / w;
    const double y = homogeneous_points.at<double>(1, column) / w;
    const double z = homogeneous_points.at<double>(2, column) / w;
    if (z <= 0.0 || z > 200.0) {
      continue;
    }

    triangulated_points.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
  }

  return triangulated_points;
}

bool ProjectWorldPoint(const cv::Point3f& point_world,
                       const cv::Point3f& camera_position,
                       const cv::Vec3f& basis_right,
                       const cv::Vec3f& basis_up,
                       const cv::Vec3f& basis_forward,
                       const ImVec2& canvas_center,
                       float focal_pixels,
                       ImVec2* projected_point) {
  const cv::Vec3f point_vector(point_world.x, point_world.y, point_world.z);
  const cv::Vec3f camera_vector(camera_position.x, camera_position.y, camera_position.z);
  const cv::Vec3f relative = point_vector - camera_vector;

  const float view_x = relative.dot(basis_right);
  const float view_y = relative.dot(basis_up);
  const float view_z = relative.dot(basis_forward);
  if (view_z <= 0.01F) {
    return false;
  }

  projected_point->x = canvas_center.x + (view_x / view_z) * focal_pixels;
  projected_point->y = canvas_center.y - (view_y / view_z) * focal_pixels;
  return true;
}

void DrawSimple3DScene(const std::vector<cv::Point3f>& gt_trajectory,
                       const std::vector<cv::Point3f>& triangulated_points,
                       float* orbit_yaw_rad,
                       float* orbit_pitch_rad,
                       float* orbit_distance,
                       cv::Point3f* orbit_target,
                       bool show_gt,
                       bool show_triangulated) {
  ImVec2 canvas_size = ImGui::GetContentRegionAvail();
  canvas_size.x = std::max(canvas_size.x, 320.0F);
  canvas_size.y = std::max(canvas_size.y, 260.0F);

  const ImVec2 canvas_min = ImGui::GetCursorScreenPos();
  const ImVec2 canvas_max(canvas_min.x + canvas_size.x, canvas_min.y + canvas_size.y);
  ImGui::InvisibleButton("3d_canvas", canvas_size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
  const bool hovered = ImGui::IsItemHovered();

  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  draw_list->AddRectFilled(canvas_min, canvas_max, IM_COL32(18, 18, 24, 255), 4.0F);
  draw_list->AddRect(canvas_min, canvas_max, IM_COL32(80, 80, 95, 255), 4.0F);

  ImGuiIO& io = ImGui::GetIO();
  if (hovered) {
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
      *orbit_yaw_rad += io.MouseDelta.x * 0.01F;
      *orbit_pitch_rad -= io.MouseDelta.y * 0.01F;
      *orbit_pitch_rad = std::clamp(*orbit_pitch_rad, -1.45F, 1.45F);
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
      orbit_target->x -= io.MouseDelta.x * 0.01F * (*orbit_distance);
      orbit_target->y += io.MouseDelta.y * 0.01F * (*orbit_distance);
    }
    if (io.MouseWheel != 0.0F) {
      *orbit_distance *= std::exp(-io.MouseWheel * 0.12F);
      *orbit_distance = std::clamp(*orbit_distance, 1.0F, 200.0F);
    }
  }

  const cv::Vec3f world_up(0.0F, 0.0F, 1.0F);
  const cv::Vec3f forward(
      std::cos(*orbit_pitch_rad) * std::cos(*orbit_yaw_rad),
      std::cos(*orbit_pitch_rad) * std::sin(*orbit_yaw_rad),
      std::sin(*orbit_pitch_rad));

  const cv::Vec3f camera_direction = cv::normalize(forward);
  const cv::Point3f camera_position(
      orbit_target->x - camera_direction[0] * (*orbit_distance),
      orbit_target->y - camera_direction[1] * (*orbit_distance),
      orbit_target->z - camera_direction[2] * (*orbit_distance));

  cv::Vec3f basis_forward = cv::normalize(cv::Vec3f(
      orbit_target->x - camera_position.x,
      orbit_target->y - camera_position.y,
      orbit_target->z - camera_position.z));
  cv::Vec3f basis_right = basis_forward.cross(world_up);
  if (cv::norm(basis_right) < 1e-5F) {
    basis_right = cv::Vec3f(1.0F, 0.0F, 0.0F);
  } else {
    basis_right = cv::normalize(basis_right);
  }
  cv::Vec3f basis_up = cv::normalize(basis_right.cross(basis_forward));

  const ImVec2 canvas_center((canvas_min.x + canvas_max.x) * 0.5F, (canvas_min.y + canvas_max.y) * 0.5F);
  const float focal_pixels = 0.75F * std::min(canvas_size.x, canvas_size.y);

  auto draw_axis = [&](const cv::Point3f& start, const cv::Point3f& end, ImU32 color) {
    ImVec2 start_2d;
    ImVec2 end_2d;
    if (!ProjectWorldPoint(start, camera_position, basis_right, basis_up, basis_forward, canvas_center, focal_pixels, &start_2d)) {
      return;
    }
    if (!ProjectWorldPoint(end, camera_position, basis_right, basis_up, basis_forward, canvas_center, focal_pixels, &end_2d)) {
      return;
    }
    draw_list->AddLine(start_2d, end_2d, color, 2.0F);
  };

  draw_axis(cv::Point3f(0.0F, 0.0F, 0.0F), cv::Point3f(1.0F, 0.0F, 0.0F), IM_COL32(255, 70, 70, 255));
  draw_axis(cv::Point3f(0.0F, 0.0F, 0.0F), cv::Point3f(0.0F, 1.0F, 0.0F), IM_COL32(80, 255, 80, 255));
  draw_axis(cv::Point3f(0.0F, 0.0F, 0.0F), cv::Point3f(0.0F, 0.0F, 1.0F), IM_COL32(80, 160, 255, 255));

  if (show_gt && gt_trajectory.size() >= 2) {
    ImVec2 previous;
    bool has_previous = false;
    for (const auto& point : gt_trajectory) {
      ImVec2 projected;
      if (!ProjectWorldPoint(point, camera_position, basis_right, basis_up, basis_forward, canvas_center, focal_pixels, &projected)) {
        has_previous = false;
        continue;
      }
      if (has_previous) {
        draw_list->AddLine(previous, projected, IM_COL32(120, 220, 255, 230), 2.0F);
      }
      previous = projected;
      has_previous = true;
    }
  }

  if (show_triangulated) {
    for (const auto& point : triangulated_points) {
      ImVec2 projected;
      if (!ProjectWorldPoint(point, camera_position, basis_right, basis_up, basis_forward, canvas_center, focal_pixels, &projected)) {
        continue;
      }
      draw_list->AddCircleFilled(projected, 2.0F, IM_COL32(255, 220, 80, 220), 6);
    }
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
  FrontendTrackingState tracking_state;
  std::vector<cv::Point3f> gt_positions_world;
  try {
    const sense8::io::EurocDatasetLoader loader;
    sequence = loader.Load(dataset_path);
    cursor.Bind(&sequence.merged_packets);
    gt_positions_world = LoadEurocGroundTruthPositions(dataset_path);

    std::string intrinsics_error;
    if (LoadEurocIntrinsics(dataset_path, &tracking_state.camera_matrix, &intrinsics_error)) {
      tracking_state.has_camera_intrinsics = true;
      spdlog::info("Loaded cam0 intrinsics for Essential-matrix RANSAC");
    } else {
      tracking_state.has_camera_intrinsics = false;
      spdlog::warn("{}", intrinsics_error);
    }
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
  ImGui::GetIO().FontGlobalScale = 1.3F;
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  double previous_time_s = glfwGetTime();
  float timeline_ratio = 0.0F;
  float ui_font_scale = 1.3F;
  float imu_window_seconds = 5.0F;
  bool show_orb_features = true;
  bool show_feature_tracks = true;
  int orb_max_features = 500;
  int max_rendered_tracks = 300;
  FrameTexture frame_texture;
  FrameTexture previous_frame_texture;
  FrameTexture dual_current_frame_texture;
  std::string frame_error;
  std::string previous_frame_error;
  std::string dual_current_frame_error;

  bool has_stable_match_pair = false;
  std::string stable_previous_image_path;
  std::string stable_current_image_path;
  std::vector<FrontendTrack> stable_inlier_tracks;
  std::vector<cv::KeyPoint> stable_previous_keypoints;
  std::vector<cv::KeyPoint> stable_current_keypoints;
  std::vector<cv::Point3f> stable_triangulated_points;

  bool show_gt_trajectory_3d = true;
  bool show_triangulated_points_3d = true;
  float orbit_yaw_rad = 0.3F;
  float orbit_pitch_rad = 0.4F;
  float orbit_distance = 8.0F;
  cv::Point3f orbit_target(0.0F, 0.0F, 0.0F);

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
    ImGui::GetIO().FontGlobalScale = ui_font_scale;

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

        ImGui::SliderFloat("UI Font Scale", &ui_font_scale, 1.0F, 2.0F, "%.2f");
        ImGui::Checkbox("Show ORB features", &show_orb_features);
        ImGui::SameLine();
        ImGui::Checkbox("Show feature tracks", &show_feature_tracks);
        ImGui::SliderInt("ORB max features", &orb_max_features, 100, 2000);
        ImGui::SliderInt("Max rendered tracks", &max_rendered_tracks, 50, 1000);

        const bool has_frame_texture = UpdateFrameTexture(
            sequence.camera_frames,
            cursor.current_timestamp_ns(),
            show_orb_features,
            orb_max_features,
          show_feature_tracks,
          &tracking_state,
            &frame_texture,
            &frame_error);

        if (has_frame_texture && tracking_state.has_match_pair) {
          has_stable_match_pair = true;
          stable_previous_image_path = tracking_state.match_previous_image_path;
          stable_current_image_path = tracking_state.match_current_image_path;
          stable_inlier_tracks = tracking_state.inlier_tracks;
          stable_previous_keypoints = tracking_state.match_previous_keypoints;
          stable_current_keypoints = tracking_state.match_current_keypoints;
          stable_triangulated_points = TriangulateDebugPoints(tracking_state, stable_inlier_tracks, 400);
        }

        ImGui::SeparatorText("Camera View");
        if (has_frame_texture) {
          const float available_width = ImGui::GetContentRegionAvail().x;
          if (ImGui::BeginTabBar("replay_view_tabs")) {
            if (ImGui::BeginTabItem("Single View")) {
              const float aspect_ratio = static_cast<float>(frame_texture.height) / static_cast<float>(frame_texture.width);
              const float image_width = std::max(320.0F, available_width);
              const float image_height = image_width * aspect_ratio;
              const ImVec2 image_top_left = ImGui::GetCursorScreenPos();
              const ImVec2 image_bottom_right(image_top_left.x + image_width, image_top_left.y + image_height);
              ImGui::Image((ImTextureID)(intptr_t)frame_texture.texture_id, ImVec2(image_width, image_height));

              if (show_feature_tracks && !tracking_state.inlier_tracks.empty()) {
                const float scale_x = image_width / static_cast<float>(frame_texture.width);
                const float scale_y = image_height / static_cast<float>(frame_texture.height);
                ImDrawList* draw_list = ImGui::GetWindowDrawList();

                const std::size_t rendered_tracks = std::min(
                    tracking_state.inlier_tracks.size(),
                    static_cast<std::size_t>(std::max(1, max_rendered_tracks)));
                for (std::size_t track_index = 0; track_index < rendered_tracks; ++track_index) {
                  const auto& track = tracking_state.inlier_tracks[track_index];
                  const ImVec2 previous_point(
                      image_top_left.x + track.previous_point.x * scale_x,
                      image_top_left.y + track.previous_point.y * scale_y);
                  const ImVec2 current_point(
                      image_top_left.x + track.current_point.x * scale_x,
                      image_top_left.y + track.current_point.y * scale_y);

                  if (current_point.x < image_top_left.x || current_point.x > image_bottom_right.x ||
                      current_point.y < image_top_left.y || current_point.y > image_bottom_right.y) {
                    continue;
                  }

                  draw_list->AddLine(previous_point, current_point, IM_COL32(230, 50, 50, 230), 1.8F);
                  draw_list->AddCircleFilled(current_point, 2.2F, IM_COL32(255, 220, 30, 220), 6);
                }
              }

              if (show_orb_features && !frame_texture.orb_keypoints.empty()) {
                const float scale_x = image_width / static_cast<float>(frame_texture.width);
                const float scale_y = image_height / static_cast<float>(frame_texture.height);
                ImDrawList* draw_list = ImGui::GetWindowDrawList();

                int hovered_keypoint_index = -1;
                float hovered_distance_sq = std::numeric_limits<float>::max();
                const ImVec2 mouse_pos = ImGui::GetIO().MousePos;
                const bool mouse_over_image = ImGui::IsMouseHoveringRect(image_top_left, image_bottom_right);

                constexpr float kHoverRadiusPixels = 8.0F;
                const float max_hover_distance_sq = kHoverRadiusPixels * kHoverRadiusPixels;

                constexpr std::size_t kMaxRenderedOverlayPoints = 1000;
                const std::size_t rendered_count = std::min(frame_texture.orb_keypoints.size(), kMaxRenderedOverlayPoints);

                for (std::size_t keypoint_index = 0; keypoint_index < rendered_count; ++keypoint_index) {
                  const auto& keypoint = frame_texture.orb_keypoints[keypoint_index];
                  const ImVec2 point(
                      image_top_left.x + keypoint.pt.x * scale_x,
                      image_top_left.y + keypoint.pt.y * scale_y);
                  draw_list->AddCircleFilled(point, 3.0F, IM_COL32(60, 255, 60, 220), 6);

                  if (mouse_over_image) {
                    const float dx = mouse_pos.x - point.x;
                    const float dy = mouse_pos.y - point.y;
                    const float dist_sq = dx * dx + dy * dy;
                    if (dist_sq <= max_hover_distance_sq && dist_sq < hovered_distance_sq) {
                      hovered_distance_sq = dist_sq;
                      hovered_keypoint_index = static_cast<int>(keypoint_index);
                    }
                  }
                }

                if (hovered_keypoint_index >= 0) {
                  const auto& hovered_keypoint = frame_texture.orb_keypoints[static_cast<std::size_t>(hovered_keypoint_index)];
                  const float u_center = hovered_keypoint.pt.x / static_cast<float>(frame_texture.width);
                  const float v_center = hovered_keypoint.pt.y / static_cast<float>(frame_texture.height);
                  const float half_u = (15.0F * 0.5F) / static_cast<float>(frame_texture.width);
                  const float half_v = (15.0F * 0.5F) / static_cast<float>(frame_texture.height);

                  const ImVec2 uv0(std::clamp(u_center - half_u, 0.0F, 1.0F), std::clamp(v_center - half_v, 0.0F, 1.0F));
                  const ImVec2 uv1(std::clamp(u_center + half_u, 0.0F, 1.0F), std::clamp(v_center + half_v, 0.0F, 1.0F));

                  ImGui::BeginTooltip();
                  ImGui::Text("ORB feature #%d", hovered_keypoint_index);
                  ImGui::Text("x=%.1f, y=%.1f", hovered_keypoint.pt.x, hovered_keypoint.pt.y);
                  ImGui::Text("response=%.3f", hovered_keypoint.response);
                  ImGui::Separator();
                  ImGui::Text("15x15 patch (zoomed)");
                  ImGui::Image((ImTextureID)(intptr_t)frame_texture.texture_id, ImVec2(180.0F, 180.0F), uv0, uv1);
                  const ImVec2 patch_min = ImGui::GetItemRectMin();
                  const ImVec2 patch_max = ImGui::GetItemRectMax();
                  const ImVec2 patch_center((patch_min.x + patch_max.x) * 0.5F, (patch_min.y + patch_max.y) * 0.5F);
                  ImDrawList* tooltip_draw_list = ImGui::GetWindowDrawList();
                  tooltip_draw_list->AddLine(
                      ImVec2(patch_center.x - 8.0F, patch_center.y - 8.0F),
                      ImVec2(patch_center.x + 8.0F, patch_center.y + 8.0F),
                      IM_COL32(255, 80, 80, 255),
                      2.0F);
                  tooltip_draw_list->AddLine(
                      ImVec2(patch_center.x - 8.0F, patch_center.y + 8.0F),
                      ImVec2(patch_center.x + 8.0F, patch_center.y - 8.0F),
                      IM_COL32(255, 80, 80, 255),
                      2.0F);
                  ImGui::EndTooltip();
                }
              }

              ImGui::TextWrapped("Image: %s", frame_texture.loaded_image_path.c_str());
              ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Dual View")) {
              if (!has_stable_match_pair) {
                ImGui::Text("Need consecutive frames for dual-view correspondence");
              } else {
                const bool has_previous_texture = UpdateTextureFromImagePath(
                    stable_previous_image_path,
                    &previous_frame_texture,
                    &previous_frame_error);
                const bool has_current_texture = UpdateTextureFromImagePath(
                    stable_current_image_path,
                    &dual_current_frame_texture,
                    &dual_current_frame_error);

                if (has_previous_texture && has_current_texture) {
                  const float image_gap = 16.0F;
                  const float pane_width = std::max(260.0F, (available_width - image_gap) * 0.5F);
                  const float previous_aspect = static_cast<float>(previous_frame_texture.height) /
                                                static_cast<float>(previous_frame_texture.width);
                  const float current_aspect = static_cast<float>(dual_current_frame_texture.height) /
                                               static_cast<float>(dual_current_frame_texture.width);
                  const float previous_height = pane_width * previous_aspect;
                  const float current_height = pane_width * current_aspect;
                  const float pane_height = std::max(previous_height, current_height);

                  const ImVec2 previous_top_left = ImGui::GetCursorScreenPos();
                  ImGui::Image((ImTextureID)(intptr_t)previous_frame_texture.texture_id, ImVec2(pane_width, previous_height));

                  ImGui::SameLine(0.0F, image_gap);
                  const ImVec2 current_top_left = ImGui::GetCursorScreenPos();
                  ImGui::Image((ImTextureID)(intptr_t)dual_current_frame_texture.texture_id, ImVec2(pane_width, current_height));

                  const ImVec2 previous_bottom_right(previous_top_left.x + pane_width, previous_top_left.y + previous_height);
                  const ImVec2 current_bottom_right(current_top_left.x + pane_width, current_top_left.y + current_height);

                  if (show_feature_tracks && !stable_inlier_tracks.empty()) {
                    const float prev_scale_x = pane_width / static_cast<float>(previous_frame_texture.width);
                    const float prev_scale_y = previous_height / static_cast<float>(previous_frame_texture.height);
                    const float curr_scale_x = pane_width / static_cast<float>(dual_current_frame_texture.width);
                    const float curr_scale_y = current_height / static_cast<float>(dual_current_frame_texture.height);
                    ImDrawList* draw_list = ImGui::GetWindowDrawList();

                    const std::size_t rendered_tracks = std::min(
                        stable_inlier_tracks.size(),
                        static_cast<std::size_t>(std::max(1, max_rendered_tracks)));
                    for (std::size_t track_index = 0; track_index < rendered_tracks; ++track_index) {
                      const auto& track = stable_inlier_tracks[track_index];
                      const ImVec2 previous_point(
                          previous_top_left.x + track.previous_point.x * prev_scale_x,
                          previous_top_left.y + track.previous_point.y * prev_scale_y);
                      const ImVec2 current_point(
                          current_top_left.x + track.current_point.x * curr_scale_x,
                          current_top_left.y + track.current_point.y * curr_scale_y);

                      draw_list->AddLine(previous_point, current_point, IM_COL32(30, 170, 255, 200), 1.0F);
                      draw_list->AddCircleFilled(previous_point, 2.0F, IM_COL32(255, 220, 30, 220), 6);
                      draw_list->AddCircleFilled(current_point, 2.0F, IM_COL32(255, 220, 30, 220), 6);
                    }
                  }

                  if (show_orb_features) {
                    const float prev_scale_x = pane_width / static_cast<float>(previous_frame_texture.width);
                    const float prev_scale_y = previous_height / static_cast<float>(previous_frame_texture.height);
                    const float curr_scale_x = pane_width / static_cast<float>(dual_current_frame_texture.width);
                    const float scale_y = current_height / static_cast<float>(dual_current_frame_texture.height);
                    ImDrawList* draw_list = ImGui::GetWindowDrawList();

                    int hovered_prev_index = -1;
                    int hovered_curr_index = -1;
                    float hovered_prev_distance_sq = std::numeric_limits<float>::max();
                    float hovered_curr_distance_sq = std::numeric_limits<float>::max();
                    const ImVec2 mouse_pos = ImGui::GetIO().MousePos;
                    const bool mouse_over_previous = ImGui::IsMouseHoveringRect(previous_top_left, previous_bottom_right);
                    const bool mouse_over_image = ImGui::IsMouseHoveringRect(current_top_left, current_bottom_right);

                    constexpr float kHoverRadiusPixels = 8.0F;
                    const float max_hover_distance_sq = kHoverRadiusPixels * kHoverRadiusPixels;

                    constexpr std::size_t kMaxRenderedOverlayPoints = 1000;
                    const std::size_t rendered_prev = std::min(stable_previous_keypoints.size(), kMaxRenderedOverlayPoints);
                    const std::size_t rendered_curr = std::min(stable_current_keypoints.size(), kMaxRenderedOverlayPoints);

                    for (std::size_t keypoint_index = 0; keypoint_index < rendered_prev; ++keypoint_index) {
                      const auto& keypoint = stable_previous_keypoints[keypoint_index];
                      const ImVec2 point(
                          previous_top_left.x + keypoint.pt.x * prev_scale_x,
                          previous_top_left.y + keypoint.pt.y * prev_scale_y);
                      draw_list->AddCircleFilled(point, 3.0F, IM_COL32(60, 255, 60, 220), 6);

                      if (mouse_over_previous) {
                        const float dx = mouse_pos.x - point.x;
                        const float dy = mouse_pos.y - point.y;
                        const float dist_sq = dx * dx + dy * dy;
                        if (dist_sq <= max_hover_distance_sq && dist_sq < hovered_prev_distance_sq) {
                          hovered_prev_distance_sq = dist_sq;
                          hovered_prev_index = static_cast<int>(keypoint_index);
                        }
                      }
                    }

                    for (std::size_t keypoint_index = 0; keypoint_index < rendered_curr; ++keypoint_index) {
                      const auto& keypoint = stable_current_keypoints[keypoint_index];
                      const ImVec2 point(
                          current_top_left.x + keypoint.pt.x * curr_scale_x,
                          current_top_left.y + keypoint.pt.y * scale_y);
                      draw_list->AddCircleFilled(point, 3.0F, IM_COL32(60, 255, 60, 220), 6);

                      if (mouse_over_image) {
                        const float dx = mouse_pos.x - point.x;
                        const float dy = mouse_pos.y - point.y;
                        const float dist_sq = dx * dx + dy * dy;
                        if (dist_sq <= max_hover_distance_sq && dist_sq < hovered_curr_distance_sq) {
                          hovered_curr_distance_sq = dist_sq;
                          hovered_curr_index = static_cast<int>(keypoint_index);
                        }
                      }
                    }

                    if (hovered_prev_index >= 0) {
                      const auto& hovered_keypoint = stable_previous_keypoints[static_cast<std::size_t>(hovered_prev_index)];
                      const float u_center = hovered_keypoint.pt.x / static_cast<float>(previous_frame_texture.width);
                      const float v_center = hovered_keypoint.pt.y / static_cast<float>(previous_frame_texture.height);
                      const float half_u = (15.0F * 0.5F) / static_cast<float>(previous_frame_texture.width);
                      const float half_v = (15.0F * 0.5F) / static_cast<float>(previous_frame_texture.height);

                      const ImVec2 uv0(std::clamp(u_center - half_u, 0.0F, 1.0F), std::clamp(v_center - half_v, 0.0F, 1.0F));
                      const ImVec2 uv1(std::clamp(u_center + half_u, 0.0F, 1.0F), std::clamp(v_center + half_v, 0.0F, 1.0F));

                      ImGui::BeginTooltip();
                      ImGui::Text("Previous frame ORB feature #%d", hovered_prev_index);
                      ImGui::Text("x=%.1f, y=%.1f", hovered_keypoint.pt.x, hovered_keypoint.pt.y);
                      ImGui::Text("response=%.3f", hovered_keypoint.response);
                      ImGui::Separator();
                      ImGui::Text("15x15 patch (zoomed)");
                      ImGui::Image((ImTextureID)(intptr_t)previous_frame_texture.texture_id, ImVec2(180.0F, 180.0F), uv0, uv1);
                      const ImVec2 patch_min = ImGui::GetItemRectMin();
                      const ImVec2 patch_max = ImGui::GetItemRectMax();
                      const ImVec2 patch_center((patch_min.x + patch_max.x) * 0.5F, (patch_min.y + patch_max.y) * 0.5F);
                      ImDrawList* tooltip_draw_list = ImGui::GetWindowDrawList();
                      tooltip_draw_list->AddLine(
                          ImVec2(patch_center.x - 8.0F, patch_center.y - 8.0F),
                          ImVec2(patch_center.x + 8.0F, patch_center.y + 8.0F),
                          IM_COL32(255, 80, 80, 255),
                          2.0F);
                      tooltip_draw_list->AddLine(
                          ImVec2(patch_center.x - 8.0F, patch_center.y + 8.0F),
                          ImVec2(patch_center.x + 8.0F, patch_center.y - 8.0F),
                          IM_COL32(255, 80, 80, 255),
                          2.0F);
                      ImGui::EndTooltip();
                    } else if (hovered_curr_index >= 0) {
                      const auto& hovered_keypoint = stable_current_keypoints[static_cast<std::size_t>(hovered_curr_index)];
                      const float u_center = hovered_keypoint.pt.x / static_cast<float>(dual_current_frame_texture.width);
                      const float v_center = hovered_keypoint.pt.y / static_cast<float>(dual_current_frame_texture.height);
                      const float half_u = (15.0F * 0.5F) / static_cast<float>(dual_current_frame_texture.width);
                      const float half_v = (15.0F * 0.5F) / static_cast<float>(dual_current_frame_texture.height);

                      const ImVec2 uv0(std::clamp(u_center - half_u, 0.0F, 1.0F), std::clamp(v_center - half_v, 0.0F, 1.0F));
                      const ImVec2 uv1(std::clamp(u_center + half_u, 0.0F, 1.0F), std::clamp(v_center + half_v, 0.0F, 1.0F));

                      ImGui::BeginTooltip();
                      ImGui::Text("Current frame ORB feature #%d", hovered_curr_index);
                      ImGui::Text("x=%.1f, y=%.1f", hovered_keypoint.pt.x, hovered_keypoint.pt.y);
                      ImGui::Text("response=%.3f", hovered_keypoint.response);
                      ImGui::Separator();
                      ImGui::Text("15x15 patch (zoomed)");
                      ImGui::Image((ImTextureID)(intptr_t)dual_current_frame_texture.texture_id, ImVec2(180.0F, 180.0F), uv0, uv1);
                      const ImVec2 patch_min = ImGui::GetItemRectMin();
                      const ImVec2 patch_max = ImGui::GetItemRectMax();
                      const ImVec2 patch_center((patch_min.x + patch_max.x) * 0.5F, (patch_min.y + patch_max.y) * 0.5F);
                      ImDrawList* tooltip_draw_list = ImGui::GetWindowDrawList();
                      tooltip_draw_list->AddLine(
                          ImVec2(patch_center.x - 8.0F, patch_center.y - 8.0F),
                          ImVec2(patch_center.x + 8.0F, patch_center.y + 8.0F),
                          IM_COL32(255, 80, 80, 255),
                          2.0F);
                      tooltip_draw_list->AddLine(
                          ImVec2(patch_center.x - 8.0F, patch_center.y + 8.0F),
                          ImVec2(patch_center.x + 8.0F, patch_center.y - 8.0F),
                          IM_COL32(255, 80, 80, 255),
                          2.0F);
                      ImGui::EndTooltip();
                    }
                  }

                  ImGui::Dummy(ImVec2(0.0F, std::max(0.0F, pane_height - std::min(previous_height, current_height))));
                  ImGui::TextWrapped("Previous: %s", previous_frame_texture.loaded_image_path.c_str());
                  ImGui::TextWrapped("Current: %s", dual_current_frame_texture.loaded_image_path.c_str());
                } else {
                  if (!has_previous_texture) {
                    ImGui::TextWrapped("%s", previous_frame_error.c_str());
                  }
                  if (!has_current_texture) {
                    ImGui::TextWrapped("%s", dual_current_frame_error.c_str());
                  }
                }
              }
              ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
          }

          ImGui::Text("ORB keypoints: %zu", frame_texture.orb_keypoints.size());
          if (frame_texture.orb_keypoints.size() > 1000) {
            ImGui::Text("Rendering first %d keypoints for UI stability", 1000);
          }
          ImGui::Text("ORB detect time: %.2f ms", frame_texture.orb_compute_ms);

          ImGui::SeparatorText("Frontend Health");
          ImGui::Text("Geometric model: %s", tracking_state.metrics.geometric_model.c_str());
          ImGui::Text("Mutual matches: %d", tracking_state.metrics.mutual_matches);
          ImGui::Text("RANSAC inliers: %d", tracking_state.metrics.inlier_matches);
          const double inlier_ratio = tracking_state.metrics.mutual_matches > 0
                                          ? static_cast<double>(tracking_state.metrics.inlier_matches) /
                                                static_cast<double>(tracking_state.metrics.mutual_matches)
                                          : 0.0;
          ImGui::Text("Inlier ratio: %.3f", inlier_ratio);
          ImGui::Text("Detect/compute: %.2f ms", tracking_state.metrics.detect_compute_ms);
          ImGui::Text("Matching: %.2f ms", tracking_state.metrics.match_ms);
          ImGui::Text("RANSAC: %.2f ms", tracking_state.metrics.ransac_ms);
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

      if (ImGui::BeginTabItem("3D View")) {
        ImGui::Text("Visualization-first scaffold (GT + debug triangulation)");
        ImGui::Checkbox("Show GT trajectory", &show_gt_trajectory_3d);
        ImGui::SameLine();
        ImGui::Checkbox("Show triangulated points", &show_triangulated_points_3d);
        ImGui::Text("GT samples: %zu", gt_positions_world.size());
        ImGui::Text("Triangulated points: %zu", stable_triangulated_points.size());
        ImGui::Text("Controls: Right-drag orbit, Middle-drag pan, Mouse wheel zoom");

        DrawSimple3DScene(
            gt_positions_world,
            stable_triangulated_points,
            &orbit_yaw_rad,
            &orbit_pitch_rad,
            &orbit_distance,
            &orbit_target,
            show_gt_trajectory_3d,
            show_triangulated_points_3d);

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
  DestroyTexture(&previous_frame_texture);
  DestroyTexture(&dual_current_frame_texture);

  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
