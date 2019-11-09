// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#pragma once

#include "energy_model/distortion_kernel/distortion_kernel_2d.h"

namespace distortion_kernel {

class SymmetricDirichletKernel2D : public DistortionKernel2D {
 public:
  Eigen::Vector2d ComputeKernelEnergy(const Eigen::Vector2d& s, bool is_prev_flipped = false) override;
  Eigen::Vector2d ComputeKernelGradient(const Eigen::Vector2d& s, bool is_prev_flipped = false) override;
  Eigen::Matrix2d ComputeKernelHessian(const Eigen::Vector2d& s, bool is_prev_flipped = false) override;

  Eigen::Vector2d GetStretchPairEigenValues(const Eigen::Vector2d& s, bool is_prev_flipped = false) override;
  void GetHessianEigenValues(const Eigen::Vector2d& s,
                             Eigen::Matrix2d* eigen_values,
                             Eigen::Matrix2d* eigen_vectors, bool is_prev_flipped = false) override;
};

}  // namespace distortion_kernel
