// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#include "stdafx.h"
#include "energy_model/distortion_kernel/symmetric_dirichlet_kernel_2d.h"

namespace distortion_kernel {

namespace {
constexpr double kSingularThreshold = 1e-8;
}  // namespace

Eigen::Vector2d SymmetricDirichletKernel2D::ComputeKernelEnergy(
    const Eigen::Vector2d& s, bool was_valid) {
  if (s[0] < kSingularThreshold || s[1] < kSingularThreshold) {
    return Eigen::Vector2d(0.0, 1.0);
  }

  double energy =
      s[0] * s[0] + s[1] * s[1] + 1.0 / (s[0] * s[0]) + 1.0 / (s[1] * s[1]);

  return Eigen::Vector2d(energy, 0.0);
}

Eigen::Vector2d SymmetricDirichletKernel2D::ComputeKernelGradient(
    const Eigen::Vector2d& s, bool was_valid) {
  Eigen::Vector2d gradient(2.0 * (s[0] - 1.0 / (s[0] * s[0] * s[0])),
                           2.0 * (s[1] - 1.0 / (s[1] * s[1] * s[1])));

  return gradient;
}

Eigen::Matrix2d SymmetricDirichletKernel2D::ComputeKernelHessian(
    const Eigen::Vector2d& s, bool was_valid) {
  Eigen::Matrix2d hessian =
      (Eigen::Matrix2d() << 2.0 + 6.0 / (s[0] * s[0] * s[0] * s[0]),
       0.0,
       0.0,
       2.0 + 6.0 / (s[1] * s[1] * s[1] * s[1]))
          .finished();
  return hessian;
}

Eigen::Vector2d SymmetricDirichletKernel2D::GetStretchPairEigenValues(
    const Eigen::Vector2d& s, bool was_valid) {
  return Eigen::Vector2d(1.0 + (s[0] * s[0] + s[0] * s[1] + s[1] * s[1]) /
                                   (s[0] * s[0] * s[0] * s[1] * s[1] * s[1]),
                         1.0 - (s[0] * s[0] - s[0] * s[1] + s[1] * s[1]) /
                                   (s[0] * s[0] * s[0] * s[1] * s[1] * s[1]));
}

void SymmetricDirichletKernel2D::GetHessianEigenValues(
    const Eigen::Vector2d& s,
    Eigen::Matrix2d* eigen_values,
    Eigen::Matrix2d* eigen_vectors, bool was_valid) {
  (*eigen_values) << 2.0 + 6.0 / (s[0] * s[0] * s[0] * s[0]), 0.0, 0.0,
      2.0 + 6.0 / (s[1] * s[1] * s[1] * s[1]);
  (*eigen_vectors) << 1.0, 0.0, 0.0, 1.0;
}

}  // namespace distortion_kernel
