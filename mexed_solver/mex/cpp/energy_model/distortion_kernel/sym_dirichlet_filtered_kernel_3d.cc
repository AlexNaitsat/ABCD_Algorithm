// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il  (Alexander Naitsat)

#include "stdafx.h"
#include "energy_model/distortion_kernel/sym_dirichlet_filtered_kernel_3d.h"

namespace distortion_kernel {
	constexpr double kSingularThreshold = 1e-5;
Eigen::Vector2d SymDirichletFilteredKernel3D::ComputeKernelEnergy(
    const Eigen::Vector3d& s, bool was_valid) {
	if (!was_valid)
		return Eigen::Vector2d(energy_minimum, 0.0);

	if (s[0] < kSingularThreshold || s[1] < kSingularThreshold || s[2] < kSingularThreshold) {
		return Eigen::Vector2d(energy_minimum, 1.0);
	}

    double energy = s[0] * s[0] + 1.0 / (s[0] * s[0]) + s[1] * s[1] +
                  1.0 / (s[1] * s[1]) + s[2] * s[2] + 1.0 / (s[2] * s[2]);

    return Eigen::Vector2d(energy, 0.0);
}

Eigen::Vector3d SymDirichletFilteredKernel3D::ComputeKernelGradient(
    const Eigen::Vector3d& s, bool was_valid) {
	if (s[2] < 0 || abs(s[1]) < kSingularThreshold || abs(s[0]) < kSingularThreshold 
		         || abs(s[2]) < kSingularThreshold) {
		return Eigen::Vector3d(0, 0,0.0);
	}
    Eigen::Vector3d gradient(2.0 * s[0] - 2.0 / (s[0] * s[0] * s[0]),
                           2.0 * s[1] - 2.0 / (s[1] * s[1] * s[1]),
                           2.0 * s[2] - 2.0 / (s[2] * s[2] * s[2]));

    return gradient;
}

Eigen::Matrix3d SymDirichletFilteredKernel3D::ComputeKernelHessian(
    const Eigen::Vector3d& s, bool was_valid) {

	if (s[2] < 0 || abs(s[1]) < kSingularThreshold || abs(s[0]) < kSingularThreshold
				 || abs(s[2]) < kSingularThreshold) {
		Eigen::Matrix3d hessian_zero;
		hessian_zero.setZero();
		return(hessian_zero);
	}

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

Eigen::VectorXd SymDirichletFilteredKernel3D::GetStretchPairEigenValues(
    const Eigen::Vector3d& s, bool was_valid) {


	if (s[2] < 0 || abs(s[1]) < kSingularThreshold || abs(s[0]) < kSingularThreshold
		|| abs(s[2]) < kSingularThreshold) {
		Eigen::VectorXd zero_output(6);

		zero_output << 0, 0, 0, 0, 0, 0;
		return zero_output;
	}

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

void SymDirichletFilteredKernel3D::GetHessianEigenValues(
    const Eigen::Vector3d& s,
    Eigen::Matrix3d* eigen_values,
    Eigen::Matrix3d* eigen_vectors,
	bool was_valid) {

	if (s[2] < 0 || abs(s[1]) < kSingularThreshold || abs(s[0]) < kSingularThreshold
				 || abs(s[2]) < kSingularThreshold) {
		(*eigen_vectors) << 1,0,0,  0,1,0,  0,0,1;
		(*eigen_values) << 0, 0, 0, 0, 0, 0, 0, 0, 0;
	}
	else {
		(*eigen_values) << 2.0 + 6.0 / (s[0] * s[0] * s[0] * s[0]), 0.0, 0.0, 0.0,
			2.0 + 6.0 / (s[1] * s[1] * s[1] * s[1]), 0.0, 0.0, 0.0,
			2.0 + 6.0 / (s[2] * s[2] * s[2] * s[2]);
		(*eigen_vectors) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
	}
}

}  // namespace distortion_kernel
