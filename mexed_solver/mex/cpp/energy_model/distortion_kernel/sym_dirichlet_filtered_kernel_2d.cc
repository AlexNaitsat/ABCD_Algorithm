// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il  (Alexander Naitsat)

#include "stdafx.h"
#include "energy_model/distortion_kernel/sym_dirichlet_filtered_kernel_2d.h"

namespace distortion_kernel {

	namespace {
		constexpr double kSingularThreshold = 1e-5;
	}  // namespace
	Eigen::Vector2d SymDirichletFilteredKernel2D::ComputeKernelEnergy(
					const Eigen::Vector2d& s, bool was_valid) {
		
		if (!was_valid)
			return Eigen::Vector2d(energy_minimum, 0.0);

		if (s[0] < kSingularThreshold || s[1] < kSingularThreshold) {
			return Eigen::Vector2d(energy_minimum, 1.0);
		}


		double energy =
			s[0] * s[0] + s[1] * s[1] + 1.0 / (s[0] * s[0]) + 1.0 / (s[1] * s[1]);

		return Eigen::Vector2d(energy, 0.0);
	}

	Eigen::Vector2d SymDirichletFilteredKernel2D::ComputeKernelGradient(
		const Eigen::Vector2d& s, bool was_valid) {
		if (s[1] < 0 || abs(s[1]) < kSingularThreshold || abs(s[0]) < kSingularThreshold) {
			return Eigen::Vector2d(0, 0.0);
		}

		Eigen::Vector2d gradient(2.0 * (s[0] - 1.0 / (s[0] * s[0] * s[0])),
			2.0 * (s[1] - 1.0 / (s[1] * s[1] * s[1])));

		return gradient;
	}

	Eigen::Matrix2d SymDirichletFilteredKernel2D::ComputeKernelHessian(
		const Eigen::Vector2d& s, bool was_valid) {
		if (s[1] < 0 ||  abs(s[1]) < kSingularThreshold || abs(s[0]) < kSingularThreshold) {
			Eigen::Matrix2d hessian_zero;
			hessian_zero.setZero();
			return(hessian_zero);
		}

		Eigen::Matrix2d hessian =
			(Eigen::Matrix2d() << 2.0 + 6.0 / (s[0] * s[0] * s[0] * s[0]),
				0.0,
				0.0,
				2.0 + 6.0 / (s[1] * s[1] * s[1] * s[1])).finished();
		return hessian;
	}

	Eigen::Vector2d SymDirichletFilteredKernel2D::GetStretchPairEigenValues(
		const Eigen::Vector2d& s, bool was_valid) {

		if (s[1] < 0 || abs(s[1]) < kSingularThreshold || abs(s[0]) < kSingularThreshold) {
			return Eigen::Vector2d(0,0);
		}

		return Eigen::Vector2d(1.0 + (s[0] * s[0] + s[0] * s[1] + s[1] * s[1]) /
			(s[0] * s[0] * s[0] * s[1] * s[1] * s[1]),
			1.0 - (s[0] * s[0] - s[0] * s[1] + s[1] * s[1]) /
			(s[0] * s[0] * s[0] * s[1] * s[1] * s[1]));
	}

	void SymDirichletFilteredKernel2D::GetHessianEigenValues(
		const Eigen::Vector2d& s,
		Eigen::Matrix2d* eigen_values,
		Eigen::Matrix2d* eigen_vectors, bool was_valid) {

		if (s[1] < 0 || abs(s[1]) < kSingularThreshold || abs(s[0]) < kSingularThreshold) {
			(*eigen_vectors) << 1, 0, 0, 1;
			(*eigen_values) << 0,0,0,0;
		}
		else {
			(*eigen_values) << 2.0 + 6.0 / (s[0] * s[0] * s[0] * s[0]), 0.0, 0.0,
				2.0 + 6.0 / (s[1] * s[1] * s[1] * s[1]);
			(*eigen_vectors) << 1.0, 0.0, 0.0, 1.0;
		}
	}

}  // namespace distortion_kernel
