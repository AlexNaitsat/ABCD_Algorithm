// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#include "stdafx.h"
#include "common/optimization/stop_check_util.h"
#include <iostream>
#include <cmath>

namespace common
{
namespace optimization
{
bool IsNumericallyConverged(const std::vector<Eigen::Vector2d> &original,
                            const std::vector<Eigen::Vector2d> &updated)
{
  double diff_norm = 0.0;
  double old_norm = 0.0;

  for (int i = 0; i < original.size(); i++)
  {
    Eigen::Vector2d diff = updated[i] - original[i];
    diff_norm += diff.transpose() * diff;
    old_norm += original[i].transpose() * original[i];
  }

  diff_norm = std::sqrt(diff_norm);
  old_norm = std::sqrt(old_norm);

  constexpr double kTolerance = 1e-10;
  constexpr double kOffset = 1e-10;

  return diff_norm < kTolerance * (kOffset + old_norm);
}

bool IsNumericallyConverged3D(const std::vector<Eigen::Vector3d> &original,
                              const std::vector<Eigen::Vector3d> &updated)
{
  double diff_norm = 0.0;
  double old_norm = 0.0;

  for (int i = 0; i < original.size(); i++)
  {
    Eigen::Vector3d diff = updated[i] - original[i];
    diff_norm += diff.transpose() * diff;
    old_norm += original[i].transpose() * original[i];
  }

  diff_norm = std::sqrt(diff_norm);
  old_norm = std::sqrt(old_norm);

  constexpr double kTolerance = 1e-10;
  constexpr double kOffset = 1e-10;

  return diff_norm < kTolerance * (kOffset + old_norm);
}
} // namespace optimization
} // namespace common
