#include <vector>

#include <gtest/gtest.h>

#include "sense8/io/replay_cursor.hpp"
#include "sense8/io/sensor_packets.hpp"

TEST(ReplayCursorTest, AdvancesThroughPacketsAndStopsAtEnd) {
  std::vector<sense8::io::SensorPacket> packets = {
      {1000, sense8::io::ImuSample{1000, {0.0, 0.0, 0.0}, {0.0, 0.0, 9.8}}},
      {2000, sense8::io::ImuSample{2000, {0.0, 0.0, 0.0}, {0.0, 0.0, 9.8}}},
      {3000, sense8::io::CameraFrame{3000, "frame.png"}},
  };

  sense8::io::ReplayCursor cursor;
  cursor.Bind(&packets);

  cursor.SetSpeed(1.0);
  cursor.SetPlaying(true);
  cursor.Update(0.0000015);

  EXPECT_EQ(cursor.packet_index(), 1);
  EXPECT_TRUE(cursor.playing());

  cursor.Update(0.0000020);
  EXPECT_EQ(cursor.packet_index(), 2);
  EXPECT_FALSE(cursor.playing());
  EXPECT_EQ(cursor.current_timestamp_ns(), 3000);
}

TEST(ReplayCursorTest, SeeksByRatio) {
  std::vector<sense8::io::SensorPacket> packets = {
      {1000, sense8::io::ImuSample{1000, {0.0, 0.0, 0.0}, {0.0, 0.0, 9.8}}},
      {2000, sense8::io::ImuSample{2000, {0.0, 0.0, 0.0}, {0.0, 0.0, 9.8}}},
      {3000, sense8::io::CameraFrame{3000, "frame.png"}},
  };

  sense8::io::ReplayCursor cursor;
  cursor.Bind(&packets);

  cursor.SeekToRatio(0.5);
  EXPECT_EQ(cursor.packet_index(), 1);

  cursor.SeekToRatio(1.0);
  EXPECT_EQ(cursor.packet_index(), 2);
}
