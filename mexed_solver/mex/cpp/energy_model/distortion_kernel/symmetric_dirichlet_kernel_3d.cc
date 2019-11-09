// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#include "energy_model/distortion_kernel/symmetric_dirichlet_kernel_3d.h"

namespace distortion_kernel {

Eigen::Vector2d SymmetricDirichletKernel3D::ComputeKernelEnergy(
    const Eigen::Vector3d& s) {
  double energy = s[0] * s[0] + 1.0 / (s[0] * s[0]) + s[1] * s[1] +
                  1.0 / (s[1] * s[1]) + s[2] * s[2] + 1.0 / (s[2] * s[2]);

  return Eigen::Vector2d(energy, 0.0);
}

Eigen::Vector3d SymmetricDirichletKernel3D::ComputeKernelGradient(
    const Eigen::Vector3d& s) {
  Eigen::Vector3d gradient(2.0 * s[0] - 2.0 / (s[0] * s[0] * s[0]),
                           2.0 * s[1] - 2.0 / (s[1] * s[1] * s[1]),
                           2.0 * s[2] - 2.0 / (s[2] * s[2] * s[2]));

  return gradient;
}

Eigen::Matrix3d SymmetricDirichletKernel3D::ComputeKernelHessian(
    const Eigen::Vector3d& s) {
  Eigen::Matrix3d hessian =
      (Eigen::Matrix3d() << 2.0 + 6.0 / (s[0] * s[0] * s[0] * s[0]),
       0.0,
       0.0,
       0.0,
       2.0 + 6.0 / (s[1] * s[1] * s[1] * s[1]),
       0.0,
       0.0,
       0.0,
       2.0 + 6.0 / (s[2] * s[2] * s[2] * s[2]))
          .finished();
  return hessian;
}

Eigen::VectorXd SymmetricDirichletKernel3D::GetStretchPairEigenValues(
    const Eigen::Vector3d& s) {
  Eigen::VectorXd output(6);
  output << 1.0 + (s[0] * s[0] + s[0] * s[1] + s[1] * s[1]) /
                      (s[0] * s[0] * s[0] * s[1] * s[1] * s[1]),
      1.0 - (s[0] * s[0] - s[0] * s[1] + s[1] * s[1]) /
                (s[0] * s[0] * s[0] * s[1] * s[1] * s[1]),
      1.0 + (s[1] * s[1] + s[1] * s[2] + s[2] * s[2]) /
                (s[1] * s[1] * s[1] * s[2] * s[2] * s[2]),
      1.0 - (s[1] * s[1] - s[1] * s[2] + s[2] * s[2]) /
                (s[1] * s[1] * s[1] * s[2] * s[2] * s[2]),
      1.0 + (s[2] * s[2] + s[2] * s[0] + s[0] * s[0]) /
                (s[2] * s[2] * s[2] * s[0] * s[0] * s[0]),
      1.0 - (s[2] * s[2] - s[2] * s[0] + s[0] * s[0]) /
                (s[2] * s[2] * s[2] * s[0] * s[0] * s[0]);
  return output;
}

void SymmetricDirichletKernel3D::GetHessianEigenValues(
    const Eigen::Vector3d& s,
    Eigen::Matrix3d* eigen_values,
    Eigen::Matrix3d* eigen_vectors) {
  (*eigen_values) << 2.0 + 6.0 / (s[0] * s[0] * s[0] * s[0]), 0.0, 0.0, 0.0,
      2.0 + 6.0 / (s[1] * s[1] * s[1] * s[1]), 0.0, 0.0, 0.0,
      2.0 + 6.0 / (s[2] * s[2] * s[2] * s[2]);
  (*eigen_vectors) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
}

}  // namespace distortion_kernel
