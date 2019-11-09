// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#include "energy_model/distortion_kernel/mips_kernel_2d.h"

namespace distortion_kernel {

namespace {
constexpr double kSingularThreshold = 1e-8;
}  // namespace

Eigen::Vector2d MIPSKernel2D::ComputeKernelEnergy(const Eigen::Vector2d& s) {
  if (s[0] < kSingularThreshold || s[1] < kSingularThreshold) {
    return Eigen::Vector2d(0.0, 1.0);
  }

  double energy = s[0] / s[1] + s[1] / s[0];

  return Eigen::Vector2d(energy, 0.0);
}

Eigen::Vector2d MIPSKernel2D::ComputeKernelGradient(const Eigen::Vector2d& s) {
  Eigen::Vector2d gradient(1.0 / s[1] - s[1] / (s[0] * s[0]),
                           1.0 / s[0] - s[0] / (s[1] * s[1]));

  return gradient;
}

Eigen::Matrix2d MIPSKernel2D::ComputeKernelHessian(const Eigen::Vector2d& s) {
  Eigen::Matrix2d hessian =
      (Eigen::Matrix2d() << 2.0 * s[1] / (s[0] * s[0] * s[0]),
       -(1.0 / (s[0] * s[0]) + 1.0 / (s[1] * s[1])),
       -(1.0 / (s[0] * s[0]) + 1.0 / (s[1] * s[1])),
       2.0 * s[0] / (s[1] * s[1] * s[1]))
          .finished();
  return hessian;
}

Eigen::Vector2d MIPSKernel2D::GetStretchPairEigenValues(
    const Eigen::Vector2d& s) {
  return Eigen::Vector2d((s[0] + s[1]) * (s[0] + s[1]),
                         -(s[0] - s[1]) * (s[0] - s[1])) *
         0.5 / (s[0] * s[0] * s[1] * s[1]);
}

void MIPSKernel2D::GetHessianEigenValues(const Eigen::Vector2d& s,
                                         Eigen::Matrix2d* eigen_values,
                                         Eigen::Matrix2d* eigen_vectors) {
  (*eigen_values) << s[0] * s[0] * s[0] * s[0] + s[1] * s[1] * s[1] * s[1] +
                         std::sqrt((s[0] * s[0] + s[1] * s[1]) *
                                   (s[0] * s[0] * s[0] * s[0] * s[0] * s[0] +
                                    s[1] * s[1] * s[1] * s[1] * s[1] * s[1])),
      0.0, 0.0,
      s[0] * s[0] * s[0] * s[0] + s[1] * s[1] * s[1] * s[1] -
          std::sqrt((s[0] * s[0] + s[1] * s[1]) *
                    (s[0] * s[0] * s[0] * s[0] * s[0] * s[0] +
                     s[1] * s[1] * s[1] * s[1] * s[1] * s[1]));

  (*eigen_vectors) << (*eigen_values)(0, 0) - 2.0 * s[0] * s[0] * s[0] * s[0],
      (*eigen_values)(1, 1) - 2.0 * s[0] * s[0] * s[0] * s[0],
      -s[0] * s[1] * (s[0] * s[0] + s[1] * s[1]),
      -s[0] * s[1] * (s[0] * s[0] + s[1] * s[1]);

  (*eigen_values) = (*eigen_values) / (s[0] * s[0] * s[0] * s[1] * s[1] * s[1]);

  double scale_1 =
      2.0 * (s[0] * s[0] + s[1] * s[1]) * (s[0] * s[0] + s[1] * s[1]) *
      (2.0 * s[0] * s[0] * s[0] * s[0] - 3.0 * s[0] * s[0] * s[1] * s[1] +
       2.0 * s[1] * s[1] * s[1] * s[1]);

  double scale_2 = 2.0 * s[0] * s[0] * s[1] * s[1] *
                   (s[0] * s[0] + s[1] * s[1]) * (s[0] * s[0] + s[1] * s[1]);

  (*eigen_vectors)(0, 0) /= std::sqrt(scale_1);
  (*eigen_vectors)(0, 1) /= std::sqrt(scale_1);
  (*eigen_vectors)(1, 0) /= std::sqrt(scale_2);
  (*eigen_vectors)(1, 1) /= std::sqrt(scale_2);
}

}  // namespace distortion_kernel
