// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#include "stdafx.h"
#include "energy_model/distortion_kernel/arap_kernel_2d.h"

namespace distortion_kernel {

namespace {
constexpr double kSingularThreshold = 1e-8;
}  // namespace

Eigen::Vector2d ARAPKernel2D::ComputeKernelEnergy(const Eigen::Vector2d& s, bool is_prev_flipped) {
  double energy = (s[0] - 1.0) * (s[0] - 1.0) + (s[1] - 1.0) * (s[1] - 1.0);

  return Eigen::Vector2d(energy, 0.0);
}

Eigen::Vector2d ARAPKernel2D::ComputeKernelGradient(const Eigen::Vector2d& s, bool is_prev_flipped) {
  Eigen::Vector2d gradient(2.0 * (s[0] - 1.0), 2.0 * (s[1] - 1.0));

  return gradient;
}

Eigen::Matrix2d ARAPKernel2D::ComputeKernelHessian(const Eigen::Vector2d& s, bool is_prev_flipped) {
  Eigen::Matrix2d hessian =
      (Eigen::Matrix2d() << 2.0, 0.0, 0.0, 2.0).finished();
  return hessian;
}

Eigen::Vector2d ARAPKernel2D::GetStretchPairEigenValues(
    const Eigen::Vector2d& s, bool is_prev_flipped) {
  double denominator = std::abs(s[0] + s[1]) > kSingularThreshold
                           ? s[0] + s[1]
                           : kSingularThreshold;
  return Eigen::Vector2d(1.0, 1.0 - 2.0 / denominator);
}

void ARAPKernel2D::GetHessianEigenValues(const Eigen::Vector2d& s,
                                         Eigen::Matrix2d* eigen_values,
                                         Eigen::Matrix2d* eigen_vectors, bool is_prev_flipped) {
  (*eigen_values) << 2.0, 0.0, 0.0, 2.0;
  (*eigen_vectors) << 1.0, 0.0, 0.0, 1.0;
}

}  // namespace distortion_kernel
