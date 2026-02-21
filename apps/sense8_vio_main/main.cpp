#include <CLI/CLI.hpp>
#include <opencv2/core.hpp>
#include <spdlog/spdlog.h>

#include "sense8/common/version.hpp"
#include "sense8/io/euroc_dataset.hpp"

int main(int argc, char** argv) {
  CLI::App app{"sense8 monocular visual-inertial pipeline entrypoint"};
  std::string dataset_path;
  std::string dataset_format = "euroc";
  app.add_option("--dataset", dataset_path, "Path to dataset root");
  app.add_option("--dataset-format", dataset_format, "Dataset format (currently: euroc)");

  CLI11_PARSE(app, argc, argv);

  spdlog::info("sense8 version: {}", sense8::common::version());
  if (!dataset_path.empty()) {
    spdlog::info("Dataset: {}", dataset_path);

    if (dataset_format != "euroc") {
      spdlog::error("Unsupported dataset format: {}", dataset_format);
      return 2;
    }

    const sense8::io::EurocDatasetLoader loader;
    const sense8::io::SensorSequence sequence = loader.Load(dataset_path);
    spdlog::info(
        "Loaded EuRoC sequence with {} IMU samples, {} camera frames, {} merged packets",
        sequence.imu_samples.size(),
        sequence.camera_frames.size(),
        sequence.merged_packets.size());

    if (!sequence.merged_packets.empty()) {
      const auto first_ts = sequence.merged_packets.front().timestamp_ns;
      const auto last_ts = sequence.merged_packets.back().timestamp_ns;
      spdlog::info("Replay timespan: {} ns", last_ts - first_ts);
    }
  }

  const cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
  spdlog::info("OpenCV initialized with matrix {}x{}", identity.rows, identity.cols);

  return 0;
}
