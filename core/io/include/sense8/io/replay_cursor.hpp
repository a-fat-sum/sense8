#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sense8/io/sensor_packets.hpp"

namespace sense8::io {

class ReplayCursor {
 public:
  ReplayCursor() = default;

  void Bind(const std::vector<SensorPacket>* packets);
  void Reset();

  void SetPlaying(bool playing);
  void TogglePlaying();
  bool playing() const;

  void SetSpeed(double speed);
  double speed() const;

  void SeekToRatio(double ratio);
  double ratio() const;

  void Update(double delta_seconds);

  std::size_t packet_index() const;
  std::int64_t current_timestamp_ns() const;
  std::int64_t start_timestamp_ns() const;
  std::int64_t end_timestamp_ns() const;

 private:
  const std::vector<SensorPacket>* packets_ = nullptr;
  std::size_t packet_index_ = 0;
  std::int64_t start_timestamp_ns_ = 0;
  std::int64_t end_timestamp_ns_ = 0;
  std::int64_t current_timestamp_ns_ = 0;
  bool playing_ = false;
  double speed_ = 1.0;
};

}  // namespace sense8::io
