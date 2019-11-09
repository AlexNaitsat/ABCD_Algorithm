// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#pragma once

#include <string>
#include <vector>

#include "Eigen/Dense"

namespace util {

void ReadObjMeshFile(std::string file_name,
                     std::vector<Eigen::Vector3d>* position,
                     std::vector<Eigen::Vector2d>* uv,
                     std::vector<Eigen::Vector3i>* mesh);

void WriteObjMeshFile(std::string file_name,
                      const std::vector<Eigen::Vector3d>& position,
                      const std::vector<Eigen::Vector2d>& uv,
                      const std::vector<Eigen::Vector3i>& mesh);

}  // namespace util
