// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.ac.il (Alexander Naitsat)
#pragma once

#include <vector>

#include <Eigen/Dense>

#include "energy_model/mesh_distortion/isotropic_svd_energy_model_2d.h"
#include "energy_model/mesh_distortion/isotropic_svd_energy_model_3d.h"

namespace common
{
namespace optimization
{
void ArmijoLineSearch(const std::vector<Eigen::Vector2d>& deformed,
                      const Eigen::VectorXd& search_direction,
                      mesh_distortion::IsotropicSVDEnergyModel2D& model,
                      std::vector<Eigen::Vector2d>* output);


double ArmijoLineSearchInBlock(const std::vector<Eigen::Vector2d>& deformed,
								const Eigen::VectorXd& search_direction,
								mesh_distortion::IsotropicSVDEnergyModel2D& model,
								std::vector<Eigen::Vector2d>* output,
								const Eigen::VectorXd& dfk,
								double fk,
								const std::vector<int>& element_block,
								const std::vector<int>& free_vertex_block,
								const data_io::SolverSpecification& solverSpec);


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


Eigen::Vector3d compute_min_step_to_singularities(const std::vector<Eigen::Vector2d>& uv, 
									 	 const std::vector<Eigen::Vector3i>& F,
										 const Eigen::VectorXd& d);

Eigen::Vector3d compute_min_step_to_singularities_inBlock(const std::vector<Eigen::Vector2d>& uv,
														const std::vector<Eigen::Vector3i>& F,
														const Eigen::VectorXd& d,
														const std::vector<int>& element_block,
														mesh_distortion::IsotropicSVDEnergyModel2D& model);

void ArmijoLineSearch3D(const std::vector<Eigen::Vector3d> &deformed,
						const Eigen::VectorXd &search_direction,
						mesh_distortion::IsotropicSVDEnergyModel3D &model,
						std::vector<Eigen::Vector3d> *output);

						
double ArmijoLineSearchInBlock3D(const std::vector<Eigen::Vector3d> &deformed,
								 const Eigen::VectorXd &search_direction,
								 mesh_distortion::IsotropicSVDEnergyModel3D &model,
								 std::vector<Eigen::Vector3d> *output,
								 const Eigen::VectorXd &dfk,
								 double fk,
								 const std::vector<int> &element_block,
								 const std::vector<int> &free_vertex_block);
} // namespace optimization
} // namespace common
