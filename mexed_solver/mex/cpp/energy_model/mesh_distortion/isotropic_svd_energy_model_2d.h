// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il (Alexander Naitsat)
#pragma once

#include <memory>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "energy_model/distortion_kernel/distortion_kernel_2d.h"

#include "common/solver/eigen/eigen_solver.h"
#ifdef _USE_PARDISO
	#include "common/solver/pardiso/pardiso_solver.h"
#endif

#include "data_io/data_io_utils.h"
namespace mesh_distortion {

class IsotropicSVDEnergyModel2D {
 public:
  void SetRestMesh( const Eigen::MatrixXd& position,
                    const std::vector<Eigen::Vector3i>& mesh);

  void SetDistortionKernel(distortion_kernel::DistortionKernel2D* kernel);

  void SetEnforcingSPD(bool enforce_spd, double spd_projection_threshold) {
    enforce_spd_ = enforce_spd;
    spd_projection_threshold_ = spd_projection_threshold;
  }
  void SetSignedSVD(bool is_signed_svd_) {
	  is_signed_svd = is_signed_svd_;
  }

  double ComputeEnergyInBlock(const std::vector<Eigen::Vector2d>& position,
							  const std::vector<int>& element_block,
							  const data_io::SolverSpecification& solverSpec,
							  bool check_invalid_elements = false);

  double ComputeEnergyInBlockParallel(const std::vector<Eigen::Vector2d>& position,
							  const std::vector<int>& element_block,
							  bool check_invalid_elements = false);

  void ComputeGradientInBlock(const std::vector<Eigen::Vector2d>& position,
						Eigen::VectorXd* gradient,
						const std::vector<int>& element_block,
						const std::vector<int>& free_vertex_block,
						data_io::SolverSpecification& solverSpec
						);
  
  void ComputeHessianNonzeroEntriesDirConstraintsInBlock(
	  const std::vector<Eigen::Vector2d>& position,
	  std::vector<Eigen::Triplet<double>>* entry_list,
	  const std::vector<int>& element_block,
	  data_io::SolverSpecification& solverSpec
  );

  void ComputeHessianNonzeroEntriesParallel( //parallel version 
	  const std::vector<Eigen::Vector2d>& position,
	  int num_threads,
	  std::vector<std::vector<Eigen::Triplet<double>>>* entry_list,
	  const std::vector<int>& element_block
  );

  std::vector<Eigen::Matrix2d> deformation_gradient_differential_;
  std::vector<Eigen::Matrix2d> svd_u_;
  std::vector<Eigen::Matrix2d> svd_v_;
  std::vector<Eigen::Vector2d> svd_s_;
  
  std::vector<common::solver::eigen::EigenSolver::EigenEntry> hessian_entries;
  int hessian_dim;
  
  Eigen::VectorXd element_distortion;
  bool is_signed_svd=false;
  bool use_flip_barrier=true;				
  std::vector<bool> is_element_valid;
  int invalid_element_num = 0;
  double ls_interval = 1.0,  
		 ls_alpha = 0.2, 
		 ls_beta = 0.8;
  int ls_max_iter = 1000;
  double prev_energy = 0; 
  double energy_difference = INFINITY;

  bool return_search_dir = false;
  bool is_flip_barrier = false;
  std::vector<bool> is_element_flip_barrier;
  std::vector<bool> is_stationary_vertex;

  std::vector<Eigen::Vector3i> mesh_;
  std::vector<int> blockFreeVertIndex;

  const std::vector<Eigen::Triplet<double>>&  GetConstraintsEntries() {
	  return constraints_entry_list;
  }
  void SetDirichletConstraintsKKT(int vertex_num, const std::vector<int>& fixed_vertices);
  void SetDirichletConstraints(int vertex_num, 
							   const std::vector<int>& free_vertices,
							   const std::vector<int>& fixed_vertices);

  void SetDirichletConstraints(int vertex_num, int value) {
	  blockFreeVertIndex.resize(vertex_num,value);
  }
  common::solver::eigen::EigenSolver  global_eigen_solver;

#ifdef _USE_PARDISO
  common::solver::pardiso::PardisoSolver global_pardiso_solver;
#endif 


  std::vector<double> volume_;
  double total_volume;
  std::vector<Eigen::Matrix2d> inverse_material_space_;
private:
  std::vector<Eigen::Matrix2d> ut_df_v;
  std::vector<double> element_energy_; //PN optimized code 
  
  std::vector<Eigen::Triplet<double>> constraints_entry_list;

  distortion_kernel::DistortionKernel2D* kernel_;

  bool enforce_spd_ = false;

  double spd_projection_threshold_ = 0.0;

  void ComputeElementHessian(const std::vector<Eigen::Vector2d>& position,
                             int element_index,
                             Eigen::Matrix<double, 6, 6>* hessian);

};

}  // namespace mesh_distortion
