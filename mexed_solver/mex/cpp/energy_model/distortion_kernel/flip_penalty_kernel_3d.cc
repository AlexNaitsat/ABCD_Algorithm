// Copyright @2019. All rights reserved.
// Authors: anaitsat@campus.technion.ac.il (Alexander Naitsat), 
//          mike323zyf@gmail.com (Yufeng Zhu)

#include "stdafx.h"
#include "energy_model/distortion_kernel/flip_penalty_kernel_3d.h"

#include <iostream>
namespace distortion_kernel {

namespace {
constexpr double kSingularThreshold = 1e-5;
}  // namespace

Eigen::Vector2d FlipPenaltyKernel3D::ComputeKernelEnergy(const Eigen::Vector3d& s_signed,
														 bool is_prev_flipped) {
	bool is_flipped = (s_signed[0]*s_signed[1]*s_signed[2]< 0);
	Eigen::Vector3d s( abs(s_signed[0]), abs(s_signed[1]), abs(s_signed[2]));

	double flip_penalty = 0, sing_penalty = 0;

	if (is_flipped)
		flip_penalty = invalidPenalty[0] + s[0]*s[1]*s[2];
	
	sing_penalty = (s[0] < kSingularThreshold)* invalidPenalty[1] +
				   (s[1] < kSingularThreshold)* invalidPenalty[1] +
			  	   (s[2] < kSingularThreshold)* invalidPenalty[1];	

	double energy = flip_penalty + sing_penalty;
	return Eigen::Vector2d(energy, 0.0);
}

Eigen::Vector3d FlipPenaltyKernel3D::ComputeKernelGradient(const Eigen::Vector3d& s_signed, bool is_prev_flipped) {
	bool is_flipped = s_signed[0] * s_signed[1] * s_signed[2] <= 0;
	Eigen::Vector3d s(abs(s_signed[0]), abs(s_signed[1]), abs(s_signed[2]));

	Eigen::Vector3d flip_gradient(0, 0,0),
					singular_gradient( -(double)(s[0] < kSingularThreshold)*sgn(s_signed[0]),
									   -(double)(s[1] < kSingularThreshold)*sgn(s_signed[1]),
									   -(double)(s[2] < kSingularThreshold)*sgn(s_signed[2]));
		;
	double flip_penalty = 0, sing_penalty = 0;

	if (is_flipped) {
		flip_gradient = Eigen::Vector3d(s[1] * s[2], s[0] * s[2], s[0] * s[1]);
	}

	Eigen::Vector3d gradient(flip_gradient + singular_gradient);
	return gradient;
}

Eigen::Matrix3d FlipPenaltyKernel3D::ComputeKernelHessian(const Eigen::Vector3d& s, bool is_prev_flipped) {
	bool is_flipped = s[0] * s[1] * s[2] <= 0;
  if (is_flipped)
	   return (Eigen::Matrix3d() << 0.0,s[2],s[1],   s[2],0.0,s[0],  s[1],s[0], 0.0).finished();
  else
	  return (Eigen::Matrix3d() << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0).finished();
}

Eigen::VectorXd FlipPenaltyKernel3D::GetStretchPairEigenValues(const Eigen::Vector3d& s, bool is_prev_flipped) {
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

void FlipPenaltyKernel3D::GetHessianEigenValues(const Eigen::Vector3d& s,
                                         Eigen::Matrix3d* eigen_values,
                                         Eigen::Matrix3d* eigen_vectors, bool is_prev_flipped) {
	//It's not correct, but I don't use it for FlipPenaltyKernel3D, because it is optimized with GD.
	bool is_flipped = std::signbit(s[0]) != std::signbit(s[1]);
	
   if (is_flipped) {
	    (*eigen_values) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
		(*eigen_vectors) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
   } else {
	   	(*eigen_values)  << 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0;
		(*eigen_vectors) << 1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0;
   }
}

}  // namespace distortion_kernel
