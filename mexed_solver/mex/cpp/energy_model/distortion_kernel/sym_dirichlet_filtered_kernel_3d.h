// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il  (Alexander Naitsat)

#pragma once

#include "energy_model/distortion_kernel/distortion_kernel_3d.h"

namespace distortion_kernel {

class SymDirichletFilteredKernel3D : public DistortionKernel3D {
 public:
  Eigen::Vector2d ComputeKernelEnergy(const Eigen::Vector3d& s, bool was_valid = false) override;
  Eigen::Vector3d ComputeKernelGradient(const Eigen::Vector3d& s, bool was_valid = false) override;
  Eigen::Matrix3d ComputeKernelHessian(const Eigen::Vector3d& s, bool was_valid = false) override;

  Eigen::VectorXd GetStretchPairEigenValues(const Eigen::Vector3d& s, bool was_valid = false) override;
  void GetHessianEigenValues(const Eigen::Vector3d& s,
                             Eigen::Matrix3d* eigen_values,
                             Eigen::Matrix3d* eigen_vectors,
							 bool was_valid = false) override;

  const double energy_minimum = 6.0;
};

}  // namespace distortion_kernel
