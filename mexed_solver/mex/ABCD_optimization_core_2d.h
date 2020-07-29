// Copyright @2019. All rights reserved.
// Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

#pragma once

#include <iostream>
#include <vector>
#include <mex.h>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include "mat.h"
#include <omp.h>
#include <chrono>
#include "common/optimization/line_search_util.h"
#include "common/optimization/stop_check_util.h"
#include "common/solver/eigen/eigen_solver.h"
#ifdef _USE_PARDISO
#include "common/solver/pardiso/pardiso_solver.h"
#endif 
#include "energy_model/mesh_distortion/isotropic_svd_energy_model_2d.h"

#include "data_io/data_io_utils.h"
#include "common/util/graph_util.h"
using namespace data_io;

//size_t  BuiltConnectivityGraphWithoutBlending(int mesh_edges_num,
//	mesh_distortion::IsotropicSVDEnergyModel2D& model,
//	SolverSpecification& solverSpec,
//	Eigen::MatrixXd& source,
//	const std::vector<Eigen::Vector2d>& target,
//	Eigen::VectorXd& search_direction,
//	const double* vv_mesh_edges,
//	double K,
//	std::vector<Eigen::Vector2i>& vv_graph_edges);

//size_t  BuiltConnectivityGraph(int mesh_edges_num, int vert_num,
//	SolverSpecification& solverSpec,
//	Eigen::MatrixXd& source,
//	const std::vector<Eigen::Vector2d>& target,
//	Eigen::VectorXd& search_direction,
//	const double* vv_mesh_edges,
//	double K,
//	std::vector<Eigen::Vector2i>& vv_graph_edges,
//	std::vector<bool>& is_stationary_v
//	);

size_t BuiltConnectivityGraphFreeBlocks(
		int mesh_edges_num,
		int vert_num,
		//mesh_distortion::IsotropicSVDEnergyModel2D& model,
		SolverSpecification& solverSpec,
		Eigen::MatrixXd& source,
		const std::vector<Eigen::Vector2d>& target,
		Eigen::VectorXd& search_direction,
		const double* vv_mesh_edges,
		double K,
		std::vector<Eigen::Vector2i>& vv_graph_edges,
		std::vector<bool>& is_stationary_v
	);

void GD_SearchDirectionInBlock(std::vector<Eigen::Vector2d>& original,
	mesh_distortion::IsotropicSVDEnergyModel2D& model,
	std::vector<Eigen::Vector2d>* updated,
	SolverSpecification& solverSpec,
	Eigen::VectorXd& gradient,
	Eigen::VectorXd& search_direction,
	const std::vector<int>& element_block,
	const std::vector<int>& free_vertex_block,
	const std::vector<int>& bnd_vertex_block);

void PN_SearchDirectionInBlock(std::vector<Eigen::Vector2d>& original,
	mesh_distortion::IsotropicSVDEnergyModel2D& model,
	std::vector<Eigen::Vector2d>* updated,
	SolverSpecification& solverSpec,
	Eigen::VectorXd& gradient,
	Eigen::VectorXd& search_direction,
	const std::vector<int>& element_block,
	const std::vector<int>& free_vertex_block,
	const std::vector<int>& bnd_vertex_block,
	common::solver::LinearSolver*  m_solver
);

void UpdateSolutionInBlock(std::vector<Eigen::Vector2d>& original,
	mesh_distortion::IsotropicSVDEnergyModel2D& model,
	std::vector<Eigen::Vector2d>* updated,
	SolverSpecification& solverSpec,
	Eigen::VectorXd& gradient,
	Eigen::VectorXd& search_direction,
	const std::vector<int>& element_block,
	const std::vector<int>& free_vertex_block,
	const std::vector<int>& bnd_vertex_block

);

void OptimizeABCD(SolverSpecification& solverSpec,
	mesh_distortion::IsotropicSVDEnergyModel2D& model,
	Eigen::VectorXd& gradient,
	Eigen::VectorXd& search_direction,
	std::vector<Eigen::Vector2d>*& original,
	std::vector<Eigen::Vector2d>*& updated,
	std::vector<Eigen::Vector2d>& updated_copy,
	const std::vector<std::vector<int>>& element_blocks,
	const std::vector<std::vector<int>>& free_vertex_blocks,
	const std::vector<std::vector<int>>& bnd_vertex_blocks,
	const std::vector<std::vector<int>>&  blocks_by_color);
