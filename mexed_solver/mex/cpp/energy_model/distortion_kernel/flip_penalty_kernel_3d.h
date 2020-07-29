// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il (Alexander Naitsat)

#pragma once

#include "energy_model/distortion_kernel/distortion_kernel_3d.h"

namespace distortion_kernel {

class FlipPenaltyKernel3D : public DistortionKernel3D {
 public:
  Eigen::Vector2d ComputeKernelEnergy(const Eigen::Vector3d& s, bool is_prev_flipped = false) override;
  Eigen::Vector3d ComputeKernelGradient(const Eigen::Vector3d& s, bool is_prev_flipped = false) override;
  Eigen::Matrix3d ComputeKernelHessian(const Eigen::Vector3d& s, bool is_prev_flipped = false) override;

  Eigen::VectorXd GetStretchPairEigenValues(const Eigen::Vector3d& s, bool is_prev_flipped = false) override;
  void GetHessianEigenValues(const Eigen::Vector3d& s,
                             Eigen::Matrix3d* eigen_values,
                             Eigen::Matrix3d* eigen_vectors, bool is_prev_flipped = false) override;

  std::vector<double> invalidPenalty;

  FlipPenaltyKernel3D(const std::vector<double>& invalidPenalty_) :
	  DistortionKernel3D(), invalidPenalty(invalidPenalty_) {}

  
  void SetModelParameter(std::string name, double value,
	  const std::vector<double>& invalidPenalty_) {
	  invalidPenalty = invalidPenalty_;

	  for (int i = 1; i < invalidPenalty_.size() - 3; i++) 
		  invalidPenalty.push_back(0);
  }

};

template <typename T> double sgn(T val) {
	return (T(0) < val) - (val < T(0));
}
}  // namespace distortion_kernel
