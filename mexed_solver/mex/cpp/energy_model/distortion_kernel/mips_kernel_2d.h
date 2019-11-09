// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#pragma once

#include "energy_model/distortion_kernel/distortion_kernel_2d.h"

namespace distortion_kernel {

class MIPSKernel2D : public DistortionKernel2D {
 public:
  Eigen::Vector2d ComputeKernelEnergy(const Eigen::Vector2d& s) override;
  Eigen::Vector2d ComputeKernelGradient(const Eigen::Vector2d& s) override;
  Eigen::Matrix2d ComputeKernelHessian(const Eigen::Vector2d& s) override;

  Eigen::Vector2d GetStretchPairEigenValues(const Eigen::Vector2d& s) override;
  void GetHessianEigenValues(const Eigen::Vector2d& s,
                             Eigen::Matrix2d* eigen_values,
                             Eigen::Matrix2d* eigen_vectors) override;
};

}  // namespace distortion_kernel
