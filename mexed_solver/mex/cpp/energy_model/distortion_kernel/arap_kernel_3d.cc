// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.ac.il (Alexander Naitsat)

#include "stdafx.h"
#include "energy_model/distortion_kernel/arap_kernel_3d.h"

namespace distortion_kernel {

namespace {
constexpr double kSingularThreshold = 1e-8;
}  // namespace

Eigen::Vector2d ARAPKernel3D::ComputeKernelEnergy(const Eigen::Vector3d& s, bool was_valid) {
  double energy = (s[0] - 1.0) * (s[0] - 1.0) + (s[1] - 1.0) * (s[1] - 1.0) +
                  (s[2] - 1.0) * (s[2] - 1.0);

  return Eigen::Vector2d(energy, 0.0);
}

Eigen::Vector3d ARAPKernel3D::ComputeKernelGradient(const Eigen::Vector3d& s, bool was_valid) {
  Eigen::Vector3d gradient(
      2.0 * (s[0] - 1.0), 2.0 * (s[1] - 1.0), 2.0 * (s[2] - 1.0));

  return gradient;
}

Eigen::Matrix3d ARAPKernel3D::ComputeKernelHessian(const Eigen::Vector3d& s, bool was_valid) {
  Eigen::Matrix3d hessian =
      (Eigen::Matrix3d() << 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)
          .finished();
  return hessian;
}

Eigen::VectorXd ARAPKernel3D::GetStretchPairEigenValues(
    const Eigen::Vector3d& s, bool was_valid) {
  double denominator_s0_s1 = std::abs(s[0] + s[1]) > kSingularThreshold
                                 ? s[0] + s[1]
                                 : kSingularThreshold;
  double denominator_s1_s2 = std::abs(s[1] + s[2]) > kSingularThreshold
                                 ? s[1] + s[2]
                                 : kSingularThreshold;
  double denominator_s2_s0 = std::abs(s[2] + s[0]) > kSingularThreshold
                                 ? s[2] + s[0]
                                 : kSingularThreshold;
  Eigen::VectorXd output(6);
  output << 1.0, 1.0 - 2.0 / denominator_s0_s1, 1.0,
      1.0 - 2.0 / denominator_s1_s2, 1.0, 1.0 - 2.0 / denominator_s2_s0;
  return output;
}

void ARAPKernel3D::GetHessianEigenValues(const Eigen::Vector3d& s,
                                         Eigen::Matrix3d* eigen_values,
                                         Eigen::Matrix3d* eigen_vectors,
										 bool was_valid) {
  (*eigen_values) << 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0;
  (*eigen_vectors) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
}

}  // namespace distortion_kernel
