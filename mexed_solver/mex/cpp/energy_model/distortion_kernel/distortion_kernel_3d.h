// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il (Alexander Naitsat)

#pragma once

#include <string>
#include <unordered_map>

#include "Eigen/Dense"

namespace distortion_kernel {

class DistortionKernel3D {
 public:
  virtual ~DistortionKernel3D() = default;
  virtual Eigen::Vector2d ComputeKernelEnergy(const Eigen::Vector3d& s, bool is_prev_flipped = false) = 0;
  virtual Eigen::Vector3d ComputeKernelGradient(const Eigen::Vector3d& s, bool is_prev_flipped = false) = 0;
  virtual Eigen::Matrix3d ComputeKernelHessian(const Eigen::Vector3d& s, bool is_prev_flipped = false) = 0;

  virtual Eigen::VectorXd GetStretchPairEigenValues(
      const Eigen::Vector3d& s, bool is_prev_flipped = false) = 0;
  virtual void GetHessianEigenValues(const Eigen::Vector3d& s,
                                     Eigen::Matrix3d* eigen_values,
                                     Eigen::Matrix3d* eigen_vectors,
									 bool is_prev_flipped = false) = 0;

  void SetModelParameter(std::string name, double value) {
    kernel_model_parameter_[name] = value;
  }

  void EnableFlipFilter(bool is_enabled) {
	  is_flip_filter = is_enabled;
  }

 protected:
  std::unordered_map<std::string, double> kernel_model_parameter_;
  bool is_flip_filter = false;
  bool is_flip_barrier = false;
};

}  // namespace distortion_kernel
