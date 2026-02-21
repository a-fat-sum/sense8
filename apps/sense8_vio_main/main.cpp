#include <CLI/CLI.hpp>
#include <opencv2/core.hpp>
#include <spdlog/spdlog.h>

#include "sense8/common/version.hpp"

int main(int argc, char** argv) {
  CLI::App app{"sense8 monocular visual-inertial pipeline entrypoint"};
  std::string dataset_path;
  app.add_option("--dataset", dataset_path, "Path to dataset root");

  CLI11_PARSE(app, argc, argv);

  spdlog::info("sense8 version: {}", sense8::common::version());
  if (!dataset_path.empty()) {
    spdlog::info("Dataset: {}", dataset_path);
  }

  const cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
  spdlog::info("OpenCV initialized with matrix {}x{}", identity.rows, identity.cols);

  return 0;
}
