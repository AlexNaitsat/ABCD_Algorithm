// Copyright @2019. All rights reserved.
// Authors: anaitsat@campus.technion.ac.il (Alexander Naitsat)
// 			mike323zyf@gmail.com (Yufeng Zhu)
#pragma once

#include <vector>

#include <Eigen/Dense>

#include "energy_model/mesh_distortion/isotropic_svd_energy_model_2d.h"
#include "energy_model/mesh_distortion/isotropic_svd_energy_model_3d.h"

namespace common
{
namespace optimization
{


double ArmijoLineSearchEnhancedInBlock(const std::vector<Eigen::Vector2d>& deformed,
										const Eigen::VectorXd& search_direction,
										mesh_distortion::IsotropicSVDEnergyModel2D& model,
										std::vector<Eigen::Vector2d>* output,
										const Eigen::VectorXd& dfk,
										double fk,
										const std::vector<int>& element_block,
										const std::vector<int>& free_vertex_block,
										const data_io::SolverSpecification& solverSpec
										);
//overloaded 3d version 
double ArmijoLineSearchEnhancedInBlock3D(const std::vector<Eigen::Vector3d>& deformed,
										const Eigen::VectorXd& search_direction,
										mesh_distortion::IsotropicSVDEnergyModel3D& model,
										std::vector<Eigen::Vector3d>* output,
										const Eigen::VectorXd& dfk,
										double fk,
										const std::vector<int>& element_block,
										const std::vector<int>& free_vertex_block,
										const data_io::SolverSpecification& solverSpec
									);


Eigen::Vector3d compute_min_step_to_singularities_inBlock(const std::vector<Eigen::Vector2d>& uv,
														const std::vector<Eigen::Vector3i>& F,
														const Eigen::VectorXd& d,
														const std::vector<int>& element_block,
														mesh_distortion::IsotropicSVDEnergyModel2D& model);

Eigen::Vector3d compute_min_step_to_singularities_3d_inBlock(const std::vector<Eigen::Vector3d>& uv,
															const std::vector<Eigen::Vector4i>& F,
															const Eigen::VectorXd& d,
															const std::vector<int>& element_block,
															mesh_distortion::IsotropicSVDEnergyModel3D& model);

					
double ArmijoLineSearchInBlock3D(const std::vector<Eigen::Vector3d> &deformed,
								 const Eigen::VectorXd &search_direction,
								 mesh_distortion::IsotropicSVDEnergyModel3D &model,
								 std::vector<Eigen::Vector3d> *output,
								 const Eigen::VectorXd &dfk,
								 double fk,
								 const std::vector<int> &element_block,
								 const std::vector<int> &free_vertex_block,
								const data_io::SolverSpecification& solverSpec);
} // namespace optimization
} // namespace common
