#include "sense8/io/replay_cursor.hpp"

#include <algorithm>

namespace sense8::io {

void ReplayCursor::Bind(const std::vector<SensorPacket>* packets) {
  packets_ = packets;
  Reset();
}

void ReplayCursor::Reset() {
  packet_index_ = 0;
  playing_ = false;

  if (packets_ == nullptr || packets_->empty()) {
    start_timestamp_ns_ = 0;
    end_timestamp_ns_ = 0;
    current_timestamp_ns_ = 0;
    return;
  }

  start_timestamp_ns_ = packets_->front().timestamp_ns;
  end_timestamp_ns_ = packets_->back().timestamp_ns;
  current_timestamp_ns_ = start_timestamp_ns_;
}

void ReplayCursor::SetPlaying(bool playing) {
  playing_ = playing;
}

void ReplayCursor::TogglePlaying() {
  playing_ = !playing_;
}

bool ReplayCursor::playing() const {
  return playing_;
}

void ReplayCursor::SetSpeed(double speed) {
  speed_ = std::max(0.01, speed);
}

double ReplayCursor::speed() const {
  return speed_;
}

void ReplayCursor::SeekToRatio(double ratio) {
  if (packets_ == nullptr || packets_->empty()) {
    return;
  }

  const double clamped = std::clamp(ratio, 0.0, 1.0);
  const double span = static_cast<double>(end_timestamp_ns_ - start_timestamp_ns_);
  current_timestamp_ns_ = start_timestamp_ns_ + static_cast<std::int64_t>(span * clamped);

  const auto it = std::lower_bound(
      packets_->begin(), packets_->end(), current_timestamp_ns_, [](const SensorPacket& packet, std::int64_t ts) {
        return packet.timestamp_ns < ts;
      });
  packet_index_ = static_cast<std::size_t>(std::distance(packets_->begin(), it));
  if (packet_index_ >= packets_->size()) {
    packet_index_ = packets_->size() - 1;
    current_timestamp_ns_ = packets_->back().timestamp_ns;
  }
}

double ReplayCursor::ratio() const {
  const auto span = end_timestamp_ns_ - start_timestamp_ns_;
  if (span <= 0) {
    return 0.0;
  }
  return static_cast<double>(current_timestamp_ns_ - start_timestamp_ns_) / static_cast<double>(span);
}

void ReplayCursor::Update(double delta_seconds) {
  if (!playing_ || packets_ == nullptr || packets_->empty()) {
    return;
  }

  const double advance_ns = delta_seconds * speed_ * 1e9;
  current_timestamp_ns_ += static_cast<std::int64_t>(advance_ns);

  while (packet_index_ + 1 < packets_->size() && (*packets_)[packet_index_ + 1].timestamp_ns <= current_timestamp_ns_) {
    ++packet_index_;
  }

  if (current_timestamp_ns_ >= end_timestamp_ns_) {
    current_timestamp_ns_ = end_timestamp_ns_;
    packet_index_ = packets_->size() - 1;
    playing_ = false;
  }
}

std::size_t ReplayCursor::packet_index() const {
  return packet_index_;
}

std::int64_t ReplayCursor::current_timestamp_ns() const {
  return current_timestamp_ns_;
}

std::int64_t ReplayCursor::start_timestamp_ns() const {
  return start_timestamp_ns_;
}

std::int64_t ReplayCursor::end_timestamp_ns() const {
  return end_timestamp_ns_;
}

}  // namespace sense8::io
