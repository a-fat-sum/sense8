#include <gtest/gtest.h>

#include "sense8/common/version.hpp"

TEST(CommonVersionTest, ReturnsNonEmptyVersion) {
  EXPECT_FALSE(sense8::common::version().empty());
}
