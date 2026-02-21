#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace sense8::io {

struct CameraFrame {
  std::int64_t timestamp_ns = 0;
  std::string image_path;
};

struct ImuSample {
  std::int64_t timestamp_ns = 0;
  double angular_velocity_rad_s[3] = {0.0, 0.0, 0.0};
  double linear_acceleration_m_s2[3] = {0.0, 0.0, 0.0};
};

using SensorPayload = std::variant<ImuSample, CameraFrame>;

struct SensorPacket {
  std::int64_t timestamp_ns = 0;
  SensorPayload payload;
};

struct SensorSequence {
  std::vector<ImuSample> imu_samples;
  std::vector<CameraFrame> camera_frames;
  std::vector<SensorPacket> merged_packets;
};

}  // namespace sense8::io
