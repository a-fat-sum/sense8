#include <filesystem>
#include <variant>

#include <gtest/gtest.h>

#include "sense8/io/euroc_dataset.hpp"
#include "sense8/io/sensor_packets.hpp"

TEST(EurocDatasetTest, LoadsAndMergesDeterministically) {
  const std::filesystem::path dataset_root = std::filesystem::path(SENSE8_TEST_DATA_DIR) / "data" / "euroc_sample";
  const sense8::io::EurocDatasetLoader loader;

  const sense8::io::SensorSequence sequence = loader.Load(dataset_root);

  ASSERT_EQ(sequence.imu_samples.size(), 3);
  ASSERT_EQ(sequence.camera_frames.size(), 2);
  ASSERT_EQ(sequence.merged_packets.size(), 5);

  EXPECT_EQ(sequence.imu_samples[0].timestamp_ns, 1000);
  EXPECT_EQ(sequence.imu_samples[1].timestamp_ns, 2000);
  EXPECT_EQ(sequence.imu_samples[2].timestamp_ns, 3000);

  EXPECT_EQ(sequence.camera_frames[0].timestamp_ns, 1500);
  EXPECT_EQ(sequence.camera_frames[1].timestamp_ns, 3500);

  EXPECT_EQ(sequence.merged_packets[0].timestamp_ns, 1000);
  EXPECT_TRUE(std::holds_alternative<sense8::io::ImuSample>(sequence.merged_packets[0].payload));

  EXPECT_EQ(sequence.merged_packets[1].timestamp_ns, 1500);
  EXPECT_TRUE(std::holds_alternative<sense8::io::CameraFrame>(sequence.merged_packets[1].payload));

  EXPECT_EQ(sequence.merged_packets[2].timestamp_ns, 2000);
  EXPECT_TRUE(std::holds_alternative<sense8::io::ImuSample>(sequence.merged_packets[2].payload));

  EXPECT_EQ(sequence.merged_packets[3].timestamp_ns, 3000);
  EXPECT_TRUE(std::holds_alternative<sense8::io::ImuSample>(sequence.merged_packets[3].payload));

  EXPECT_EQ(sequence.merged_packets[4].timestamp_ns, 3500);
  EXPECT_TRUE(std::holds_alternative<sense8::io::CameraFrame>(sequence.merged_packets[4].payload));
}
