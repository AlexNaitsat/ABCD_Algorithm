// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#pragma once

#include <memory>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "energy_model/distortion_kernel/distortion_kernel_3d.h"

namespace mesh_distortion
{

class IsotropicSVDEnergyModel3D
{
public:
  void SetRestMesh(const Eigen::MatrixXd &position,
                   const std::vector<Eigen::Vector4i> &mesh);

  void SetDistortionKernel(distortion_kernel::DistortionKernel3D *kernel);

  void SetEnforcingSPD(bool enforce_spd, double spd_projection_threshold)
  {
    enforce_spd_ = enforce_spd;
    spd_projection_threshold_ = spd_projection_threshold;
  }

  void SetSignedSVD(bool is_signed_svd_)
  {
    is_signed_svd = is_signed_svd_;
  }

  double ComputeEnergy(const std::vector<Eigen::Vector3d> &position);
  double ComputeEnergyInBlock(const std::vector<Eigen::Vector3d> &position,
                              const std::vector<int> &element_block);

  void ComputeGradient(const std::vector<Eigen::Vector3d> &position,
                       Eigen::VectorXd *gradient);
  void ComputeGradientInBlock(const std::vector<Eigen::Vector3d> &position,
                              Eigen::VectorXd *gradient,
                              const std::vector<int> &element_block,
                              const std::vector<int> &free_vertex_block
                              //const std::vector<int>& bnd_vertex_block
  );

  void ComputeHessian(const std::vector<Eigen::Vector3d> &position,
                      Eigen::SparseMatrix<double> *hessian);

  void ComputeHessianNonzeroEntries(
      const std::vector<Eigen::Vector3d> &position,
      std::vector<Eigen::Triplet<double>> *entry_list);

  void ComputeHessianNonzeroEntriesDirConstraintsInBlock(
      const std::vector<Eigen::Vector3d> &position,
      std::vector<Eigen::Triplet<double>> *entry_list,
      const std::vector<int> &element_block);

  void SetDirichletConstraints(int vertex_num, int value)
  {
    blockFreeVertIndex.resize(vertex_num, value);
  }

  std::vector<Eigen::Matrix3d> deformation_gradient_differential_;
  std::vector<Eigen::Matrix3d> ut_df_v;
  std::vector<Eigen::Matrix3d> svd_u_;
  std::vector<Eigen::Matrix3d> svd_v_;
  std::vector<Eigen::Vector3d> svd_s_;

  Eigen::VectorXd element_distortion;

  std::vector<double> volume_;
  std::vector<Eigen::Matrix3d> inverse_material_space_;
  std::vector<Eigen::Vector4i> mesh_;
  std::vector<int> blockFreeVertIndex;

  distortion_kernel::DistortionKernel3D *kernel_;

  bool enforce_spd_ = false;

  double spd_projection_threshold_ = 0.0;

  bool is_signed_svd = false;
  bool use_flip_barrier = true;

  double ls_interval = 1.0,
         ls_alpha = 0.2,
         ls_beta = 0.8;
  bool return_search_dir = false;

  void ComputeElementHessian(const std::vector<Eigen::Vector3d> &position,
                             int element_index,
                             Eigen::Matrix<double, 12, 12> *hessian);
};

} // namespace mesh_distortion
