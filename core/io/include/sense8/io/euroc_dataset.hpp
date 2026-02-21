#pragma once

#include <filesystem>

#include "sense8/io/sensor_packets.hpp"

namespace sense8::io {

class EurocDatasetLoader {
 public:
  SensorSequence Load(const std::filesystem::path& dataset_root) const;
};

}  // namespace sense8::io
