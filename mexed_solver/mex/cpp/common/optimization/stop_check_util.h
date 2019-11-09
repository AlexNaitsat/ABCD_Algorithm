// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#pragma once

#include <vector>

#include <Eigen/Dense>

namespace common
{
namespace optimization
{
bool IsNumericallyConverged(const std::vector<Eigen::Vector2d> &original,
                            const std::vector<Eigen::Vector2d> &updated);

bool IsNumericallyConverged3D(const std::vector<Eigen::Vector3d> &original,
                              const std::vector<Eigen::Vector3d> &updated);
} // namespace optimization
} // namespace common
