// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il (Alexander Naitsat)

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

bool IsNumericallyConverged3DInBlock(const std::vector<Eigen::Vector3d> &original,
									 const std::vector<Eigen::Vector3d> &updated,
									 const std::vector<int>&   free_vertex_block);

} // namespace optimization
} // namespace common
