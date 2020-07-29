// Copyright @2019. All rights reserved.
// Authors: anaitsat@campus.technion.ac.il (Alexander Naitsat),
//          mike323zyf@gmail.com (Yufeng Zhu)

#include "stdafx.h"
#include "energy_model/distortion_kernel/flip_penalty_kernel_2d.h"

#include <iostream>
namespace distortion_kernel {

namespace {
constexpr double kSingularThreshold = 1e-5;
}  // namespace

Eigen::Vector2d FlipPenaltyKernel2D::ComputeKernelEnergy(const Eigen::Vector2d& s_signed,
														 bool is_prev_flipped) {
	bool is_flipped = (s_signed[0]* s_signed[1] < 0);
	Eigen::Vector2d s( abs(s_signed[0]), abs(s_signed[1]) );

	double flip_penalty = 0, sing_penalty = 0;

	if (is_flipped)
		flip_penalty = invalidPenalty[0] + s[0]*s[1];
	
	sing_penalty = (s[0] < kSingularThreshold)* invalidPenalty[1] +
				   (s[1] < kSingularThreshold)* invalidPenalty[1];

	double energy = flip_penalty + sing_penalty;
	return Eigen::Vector2d(energy, 0.0);
}

Eigen::Vector2d FlipPenaltyKernel2D::ComputeKernelGradient(const Eigen::Vector2d& s_signed, bool is_prev_flipped) {
	bool is_flipped = s_signed[0] * s_signed[1] <= 0;
	Eigen::Vector2d s(abs(s_signed[0]), abs(s_signed[1]));

	Eigen::Vector2d flip_gradient(0, 0),
					singular_gradient( (double)(s[0] < kSingularThreshold),
									   (double)(s[1] < kSingularThreshold) );
	double flip_penalty = 0, sing_penalty = 0;

	if (is_flipped)
		flip_gradient = Eigen::Vector2d(s[1],s[0]);

	Eigen::Vector2d gradient(flip_gradient + singular_gradient);
	return gradient;
}

Eigen::Matrix2d FlipPenaltyKernel2D::ComputeKernelHessian(const Eigen::Vector2d& s, bool is_prev_flipped) {
	bool is_flipped = s[0] * s[1] <= 0;
  if (is_flipped)
      return (Eigen::Matrix2d() << 0.0, 1.0, 1.0, 0.0).finished();
  else
	  return (Eigen::Matrix2d() << 0.0, 0.0, 0.0, 0.0).finished();
}

Eigen::Vector2d FlipPenaltyKernel2D::GetStretchPairEigenValues(const Eigen::Vector2d& s, bool is_prev_flipped) {
  double denominator = std::abs(s[0] + s[1]) > kSingularThreshold
                           ? s[0] + s[1]
                           : kSingularThreshold;
  return Eigen::Vector2d(1.0, 1.0 - 2.0 / denominator);
}

void FlipPenaltyKernel2D::GetHessianEigenValues(const Eigen::Vector2d& s,
                                         Eigen::Matrix2d* eigen_values,
                                         Eigen::Matrix2d* eigen_vectors, bool is_prev_flipped) {
	bool is_flipped = std::signbit(s[0]) != std::signbit(s[1]);
	
   if (is_flipped) {
		(*eigen_values)  << -1.0, 0.0, 0.0, 1.0;
		(*eigen_vectors) << -1.0,  1.0, 1.0,  1.0;
   } else {
	   	(*eigen_values)  << 0.0, 0.0, 0.0, 0.0; 
		(*eigen_vectors) << 1.0, 0.0, 0.0, 1.0;
   }
}

}  // namespace distortion_kernel
