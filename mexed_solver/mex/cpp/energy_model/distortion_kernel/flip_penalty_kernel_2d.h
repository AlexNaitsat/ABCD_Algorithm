// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu),
// anaitsat@campus.technion.ac.il (Alexander Naitsat)

#pragma once

#include "energy_model/distortion_kernel/distortion_kernel_2d.h"

namespace distortion_kernel {

class FlipPenaltyKernel2D : public DistortionKernel2D {
 public:
  Eigen::Vector2d ComputeKernelEnergy(const Eigen::Vector2d& s, bool is_prev_flipped = false) override;
  Eigen::Vector2d ComputeKernelGradient(const Eigen::Vector2d& s, bool is_prev_flipped = false) override;
  Eigen::Matrix2d ComputeKernelHessian(const Eigen::Vector2d& s, bool is_prev_flipped = false) override;

  Eigen::Vector2d GetStretchPairEigenValues(const Eigen::Vector2d& s, bool is_prev_flipped = false) override;
  void GetHessianEigenValues(const Eigen::Vector2d& s,
                             Eigen::Matrix2d* eigen_values,
                             Eigen::Matrix2d* eigen_vectors, bool is_prev_flipped = false) override;

  std::vector<double> invalidPenalty;

  FlipPenaltyKernel2D(const std::vector<double>& invalidPenalty_) :
	  DistortionKernel2D(), invalidPenalty(invalidPenalty_) {}

  
  void SetModelParameter(std::string name, double value,
	  const std::vector<double>& invalidPenalty_) {
	  invalidPenalty = invalidPenalty_;

	  for (int i = 1; i < invalidPenalty_.size() - 3; i++) 
		  invalidPenalty.push_back(0);
  }

};

}  // namespace distortion_kernel
