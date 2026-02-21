#include "sense8/io/euroc_dataset.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

namespace sense8::io {
namespace {

std::string Trim(std::string value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c) != 0; });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) { return std::isspace(c) != 0; }).base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

bool IsCommentOrEmpty(const std::string& line) {
  const std::string trimmed = Trim(line);
  return trimmed.empty() || trimmed.starts_with("#");
}

std::vector<std::string> SplitCsv(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(Trim(field));
  }
  return fields;
}

std::int64_t ParseInt64(const std::string& value, const std::string& field_name, const std::filesystem::path& path) {
  try {
    return std::stoll(value);
  } catch (const std::exception& ex) {
    throw std::runtime_error(fmt::format("Failed to parse {} in {}: {}", field_name, path.string(), ex.what()));
  }
}

double ParseDouble(const std::string& value, const std::string& field_name, const std::filesystem::path& path) {
  try {
    return std::stod(value);
  } catch (const std::exception& ex) {
    throw std::runtime_error(fmt::format("Failed to parse {} in {}: {}", field_name, path.string(), ex.what()));
  }
}

std::vector<ImuSample> LoadImuSamples(const std::filesystem::path& imu_csv_path) {
  std::ifstream stream(imu_csv_path);
  if (!stream.is_open()) {
    throw std::runtime_error(fmt::format("Could not open IMU CSV: {}", imu_csv_path.string()));
  }

  std::vector<ImuSample> samples;
  std::string line;
  while (std::getline(stream, line)) {
    if (IsCommentOrEmpty(line)) {
      continue;
    }

    const auto fields = SplitCsv(line);
    if (fields.size() != 7 || fields[0] == "timestamp") {
      continue;
    }

    ImuSample sample;
    sample.timestamp_ns = ParseInt64(fields[0], "timestamp", imu_csv_path);
    sample.angular_velocity_rad_s[0] = ParseDouble(fields[1], "w_RS_S_x [rad s^-1]", imu_csv_path);
    sample.angular_velocity_rad_s[1] = ParseDouble(fields[2], "w_RS_S_y [rad s^-1]", imu_csv_path);
    sample.angular_velocity_rad_s[2] = ParseDouble(fields[3], "w_RS_S_z [rad s^-1]", imu_csv_path);
    sample.linear_acceleration_m_s2[0] = ParseDouble(fields[4], "a_RS_S_x [m s^-2]", imu_csv_path);
    sample.linear_acceleration_m_s2[1] = ParseDouble(fields[5], "a_RS_S_y [m s^-2]", imu_csv_path);
    sample.linear_acceleration_m_s2[2] = ParseDouble(fields[6], "a_RS_S_z [m s^-2]", imu_csv_path);
    samples.push_back(sample);
  }

  std::sort(samples.begin(), samples.end(), [](const ImuSample& lhs, const ImuSample& rhs) {
    return lhs.timestamp_ns < rhs.timestamp_ns;
  });

  return samples;
}

std::vector<CameraFrame> LoadCameraFrames(const std::filesystem::path& camera_csv_path,
                                          const std::filesystem::path& image_root) {
  std::ifstream stream(camera_csv_path);
  if (!stream.is_open()) {
    throw std::runtime_error(fmt::format("Could not open camera CSV: {}", camera_csv_path.string()));
  }

  std::vector<CameraFrame> frames;
  std::string line;
  while (std::getline(stream, line)) {
    if (IsCommentOrEmpty(line)) {
      continue;
    }

    const auto fields = SplitCsv(line);
    if (fields.size() != 2 || fields[0] == "timestamp") {
      continue;
    }

    CameraFrame frame;
    frame.timestamp_ns = ParseInt64(fields[0], "timestamp", camera_csv_path);
    frame.image_path = (image_root / fields[1]).lexically_normal().string();
    frames.push_back(std::move(frame));
  }

  std::sort(frames.begin(), frames.end(), [](const CameraFrame& lhs, const CameraFrame& rhs) {
    return lhs.timestamp_ns < rhs.timestamp_ns;
  });

  return frames;
}

std::vector<SensorPacket> MergePackets(const std::vector<ImuSample>& imu_samples,
                                       const std::vector<CameraFrame>& camera_frames) {
  std::vector<SensorPacket> packets;
  packets.reserve(imu_samples.size() + camera_frames.size());

  std::size_t imu_index = 0;
  std::size_t camera_index = 0;

  while (imu_index < imu_samples.size() || camera_index < camera_frames.size()) {
    const bool has_imu = imu_index < imu_samples.size();
    const bool has_camera = camera_index < camera_frames.size();

    const bool take_imu = has_imu && (!has_camera || imu_samples[imu_index].timestamp_ns <= camera_frames[camera_index].timestamp_ns);
    if (take_imu) {
      const ImuSample& sample = imu_samples[imu_index++];
      packets.push_back(SensorPacket{sample.timestamp_ns, sample});
      continue;
    }

    const CameraFrame& frame = camera_frames[camera_index++];
    packets.push_back(SensorPacket{frame.timestamp_ns, frame});
  }

  return packets;
}

}  // namespace

SensorSequence EurocDatasetLoader::Load(const std::filesystem::path& dataset_root) const {
  const auto imu_csv = dataset_root / "mav0" / "imu0" / "data.csv";
  const auto cam_csv = dataset_root / "mav0" / "cam0" / "data.csv";
  const auto cam_image_root = dataset_root / "mav0" / "cam0" / "data";

  SensorSequence sequence;
  sequence.imu_samples = LoadImuSamples(imu_csv);
  sequence.camera_frames = LoadCameraFrames(cam_csv, cam_image_root);
  sequence.merged_packets = MergePackets(sequence.imu_samples, sequence.camera_frames);

  return sequence;
}

}  // namespace sense8::io
