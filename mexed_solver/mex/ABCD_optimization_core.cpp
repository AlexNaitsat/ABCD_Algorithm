// Copyright @2019. All rights reserved.
// Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

#include "stdafx.h"

#include <iostream>
#include <vector>
#include <mex.h>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include "mat.h"
#include <omp.h>
#include <chrono>
#include "ABCD_optimization_core.h"
#include <time.h>

size_t  BuiltConnectivityGraphWithoutBlending(int mesh_edges_num,
	mesh_distortion::IsotropicSVDEnergyModel2D& model,
	SolverSpecification& solverSpec,
	Eigen::MatrixXd& source,
	const std::vector<Eigen::Vector2d>& target,
	Eigen::VectorXd& search_direction,
	const double* vv_mesh_edges,
	double K,
	std::vector<Eigen::Vector2i>& vv_graph_edges)
{
	vv_graph_edges.clear();
	int target_dim = target[0].size();
	for (int i = 0; i < mesh_edges_num; i++) {
		size_t u = vv_mesh_edges[i], v = vv_mesh_edges[i + mesh_edges_num];
		double d_uv_norm = 0;
		Eigen::Vector2d d_u(target_dim), d_v(target_dim);
		double d_uv_norm_aqr = 0;

		auto x_uv = target[u] - target[v];
		auto y_uv = source.row(u) - source.row(v);

		for (int j = 0; j < target_dim; j++) {
			d_u[j] = search_direction[target_dim * u + j];
			d_v[j] = search_direction[target_dim * v + j];
		}
		bool is_u_stationary = d_u.norm() <   solverSpec.zero_grad_eps,
			 is_v_stationary = d_v.norm() <   solverSpec.zero_grad_eps;

		if (is_u_stationary != is_v_stationary){
			continue;
		}
		auto d_uv = d_u - d_v;

		double x_uv_norm = x_uv.norm();
		double Luv = (x_uv_norm > 1e-16) ? d_uv.norm()*y_uv.norm() / x_uv_norm : INFINITY;

		if (Luv < K) {
			vv_graph_edges.push_back(Eigen::Vector2i(std::max(u, v), std::min(u, v)));
		}
	}
	size_t unique_graph_edges_num = vv_graph_edges.size();
	return unique_graph_edges_num;
}


size_t  BuiltConnectivityGraph(int mesh_edges_num,
							   int vert_num,
	mesh_distortion::IsotropicSVDEnergyModel2D& model,
	SolverSpecification& solverSpec,
	Eigen::MatrixXd& source,
	const std::vector<Eigen::Vector2d>& target,
	Eigen::VectorXd& search_direction,
	const double* vv_mesh_edges,
	double K,
	std::vector<Eigen::Vector2i>& vv_graph_edges)
{
	vv_graph_edges.clear();
	int target_dim = target[0].size();
	std::vector<double> Luv(mesh_edges_num,0);
	std::vector<bool>  is_stationary_vert(vert_num,false);
	double Luv_max = -1;
	for (int i = 0; i < mesh_edges_num; i++) {
		size_t u = vv_mesh_edges[i], v = vv_mesh_edges[i + mesh_edges_num];
		double d_uv_norm = 0;
		Eigen::Vector2d d_u(target_dim), d_v(target_dim);
		double d_uv_norm_aqr = 0;

		auto x_uv = target[u] - target[v];
		auto y_uv = source.row(u) - source.row(v);

		for (int j = 0; j < target_dim; j++) {
			d_u[j] = search_direction[target_dim * u + j];
			d_v[j] = search_direction[target_dim * v + j];
		}
		bool is_u_stationary = d_u.norm() < solverSpec.zero_grad_eps,
			is_v_stationary = d_v.norm() < solverSpec.zero_grad_eps;
		is_stationary_vert[u] = is_u_stationary;
		is_stationary_vert[v] = is_v_stationary;

		if (is_u_stationary != is_v_stationary){
			continue;
		}
		auto d_uv = d_u - d_v;
		double x_uv_norm = x_uv.norm();
		Luv[i] = (x_uv_norm > 1e-16) ? d_uv.norm()*y_uv.norm() / x_uv_norm : INFINITY;
		if (Luv[i] > Luv_max){
			Luv_max = Luv[i];
		}
	}

	for (int i = 0; i < mesh_edges_num; i++) {
		size_t u = vv_mesh_edges[i], v = vv_mesh_edges[i + mesh_edges_num];
		if (is_stationary_vert[u] && is_stationary_vert[v]) {
			vv_graph_edges.push_back(Eigen::Vector2i(std::max(u, v), std::min(u, v)));
		}  else {
			if ((!is_stationary_vert[u] && !is_stationary_vert[v]) &&
				(Luv[i] < solverSpec.K_hat * Luv_max * 1.0001))
			{
				vv_graph_edges.push_back(Eigen::Vector2i(std::max(u, v), std::min(u, v)));
			}
		}
	}

	size_t unique_graph_edges_num = vv_graph_edges.size();
	return unique_graph_edges_num;
}



size_t  BuiltConnectivityGraphSingleStationaryBlock(
						int mesh_edges_num,
						int vert_num,
						mesh_distortion::IsotropicSVDEnergyModel2D& model,
						SolverSpecification& solverSpec,
						Eigen::MatrixXd& source,
						const std::vector<Eigen::Vector2d>& target,
						Eigen::VectorXd& search_direction,
						const double* vv_mesh_edges,
						double K,
						std::vector<Eigen::Vector2i>& vv_graph_edges) 
{
	vv_graph_edges.clear();
	int target_dim = target[0].size();
	std::vector<double> Luv(mesh_edges_num, 0);
	std::vector<bool>  is_stationary_vert(vert_num, false);
	std::set<size_t> stationary_vertices;
	double Luv_max = -1;
	for (int i = 0; i < mesh_edges_num; i++) {
		size_t u = vv_mesh_edges[i], v = vv_mesh_edges[i + mesh_edges_num];
		double d_uv_norm = 0;
		Eigen::Vector2d d_u(target_dim), d_v(target_dim);
		double d_uv_norm_aqr = 0;

		auto x_uv = target[u] - target[v];
		auto y_uv = source.row(u) - source.row(v);

		for (int j = 0; j < target_dim; j++) {
			d_u[j] = search_direction[target_dim * u + j];
			d_v[j] = search_direction[target_dim * v + j];
		}
		bool is_u_stationary = d_u.norm() <= solverSpec.zero_grad_eps,
			is_v_stationary = d_v.norm() <= solverSpec.zero_grad_eps;
		is_stationary_vert[u] = is_u_stationary;
		is_stationary_vert[v] = is_v_stationary;
		if (is_u_stationary){
			stationary_vertices.insert(u);
		}
		if (is_v_stationary){
			stationary_vertices.insert(v);
		}

		if (is_u_stationary != is_v_stationary){
			continue;
		}

		auto d_uv = d_u - d_v;
		double x_uv_norm = x_uv.norm();
		Luv[i] = (x_uv_norm > 1e-16) ? d_uv.norm()*y_uv.norm() / x_uv_norm : INFINITY;
		if (Luv[i] > Luv_max){
			Luv_max = Luv[i];
		}
	}

	for (int i = 0; i < mesh_edges_num; i++) {
		size_t u = vv_mesh_edges[i], v = vv_mesh_edges[i + mesh_edges_num];
		if ((!is_stationary_vert[u] && !is_stationary_vert[v]) &&
			(Luv[i] < solverSpec.K_hat * Luv_max * 1.0001))
		{
			vv_graph_edges.push_back(Eigen::Vector2i(std::max(u, v), std::min(u, v)));
		}
	}

	if (stationary_vertices.size() > 2)
	{
		auto stationary_iter = stationary_vertices.begin();
		auto fist_stationary_vert = (*stationary_iter);
		stationary_iter++;
		for (; stationary_iter != stationary_vertices.end(); stationary_iter++)
		{
			vv_graph_edges.push_back(Eigen::Vector2i(fist_stationary_vert, (*stationary_iter)));
		}
	}

	size_t unique_graph_edges_num = vv_graph_edges.size();
	return unique_graph_edges_num;
}


void GD_SearchDirectionInBlock(std::vector<Eigen::Vector2d>& original,
	mesh_distortion::IsotropicSVDEnergyModel2D& model,
	std::vector<Eigen::Vector2d>* updated,
	SolverSpecification& solverSpec,
	Eigen::VectorXd& gradient,
	Eigen::VectorXd& search_direction,
	const std::vector<int>& element_block,
	const std::vector<int>& free_vertex_block,
	const std::vector<int>& bnd_vertex_block) {
	assert(updated != nullptr);

	for (auto vi : free_vertex_block) {
		search_direction[2 * vi] = -gradient[2 * vi];
		search_direction[2 * vi + 1] = -gradient[2 * vi + 1];
	}
}

void PN_SearchDirectionInBlock(std::vector<Eigen::Vector2d>& original,
	mesh_distortion::IsotropicSVDEnergyModel2D& model,
	std::vector<Eigen::Vector2d>* updated,
	SolverSpecification& solverSpec,
	Eigen::VectorXd& gradient,
	Eigen::VectorXd& search_direction,
	const std::vector<int>& element_block,
	const std::vector<int>& free_vertex_block,
	const std::vector<int>& bnd_vertex_block)
{
	assert(updated != nullptr);
	std::vector<common::solver::eigen::EigenSolver::EigenEntry> entry_list;
	int ver_num = free_vertex_block.size();

	model.ComputeHessianNonzeroEntriesDirConstraintsInBlock(original, &entry_list,
		element_block);
	if (!model.use_pardiso_solver) {
		common::solver::eigen::EigenSolver  m_solver;
		clock_t begin = clock();
		m_solver.SetPattern(entry_list, 2 * ver_num);
		m_solver.AnalyzePattern();
		clock_t middle = clock();
		m_solver.Factorize();
		m_solver.SolveDirichletConstraints(-gradient, &search_direction, free_vertex_block, bnd_vertex_block);
		clock_t end = clock();
		//std::cout << "\nTotal eigen time=" << double(end - begin) / CLOCKS_PER_SEC << std::endl;
		//	<< ", analyze-solve:" << double(end - middle) / CLOCKS_PER_SEC << std::endl;
	}
	else {
	#ifdef _USE_PARDISO
		common::solver::pardiso::PardisoSolver m_solver_pardiso;
		clock_t begin = clock();
		m_solver_pardiso.Init(-2);
		clock_t after_init = clock();
		m_solver_pardiso.SetPatter4EigenUpper(entry_list, 2 * ver_num);
		clock_t after_set_pattern = clock();
		m_solver_pardiso.AnalyzePattern();
		clock_t middle = clock();
		m_solver_pardiso.Factorize();
		m_solver_pardiso.SolveDirichletConstraints(-gradient, &search_direction, free_vertex_block, bnd_vertex_block);
		clock_t end = clock();
		std::cout << "\nTotal pardiso time=" << double(end - begin) / CLOCKS_PER_SEC
			<< ",analyze-solve:" << double(end - middle) / CLOCKS_PER_SEC
			<< " ,init:" << double(after_init - begin) / CLOCKS_PER_SEC
			<< " ,SetPattern:" << double(after_set_pattern - after_init) / CLOCKS_PER_SEC
			<< " ,AnalyzePattern:" << double(middle - after_set_pattern) / CLOCKS_PER_SEC
			<< std::endl;
	#endif 
	}
}

void UpdateSolutionInBlock(std::vector<Eigen::Vector2d>& original,
	mesh_distortion::IsotropicSVDEnergyModel2D& model,
	std::vector<Eigen::Vector2d>* updated,
	SolverSpecification& solverSpec,
	Eigen::VectorXd& gradient,
	Eigen::VectorXd& search_direction,
	const std::vector<int>& element_block,
	const std::vector<int>& free_vertex_block,
	const std::vector<int>& bnd_vertex_block)
{

	if (!free_vertex_block.size()) {
		return;
	}

	int ver_num = free_vertex_block.size();
	for (auto vi : bnd_vertex_block) {
		search_direction(2 * vi) = 0;
		search_direction(2 * vi + 1) = 0;
		(*updated)[vi] = original[vi];
		model.blockFreeVertIndex[vi] = -1;
	}

	for (size_t i = 0; i < ver_num; i++) {
		int v = free_vertex_block[i];
		model.blockFreeVertIndex[v] = i;
	}

	for (int block_iter = 0; block_iter < solverSpec.max_block_iterations; block_iter++) {
		double fk = 0, grad_sq_norm;
		if (solverSpec.is_distortion_data_updated) {
			//std::cout << "\n distortion data is updated  !!!";
			fk = model.prev_energy;
			solverSpec.is_distortion_data_updated = false;
		} else {
				fk = model.ComputeEnergyInBlock(original, element_block, true);
				double grad_sq_norm = 0;
				model.ComputeGradientInBlock(original, &gradient, element_block, free_vertex_block);
				for (auto vi : free_vertex_block) {
					double vi_gradient[] = { gradient[2 * vi] ,gradient[2 * vi + 1] };
					grad_sq_norm += vi_gradient[0] * vi_gradient[0] + vi_gradient[1] * vi_gradient[1];
					if (mxIsNaN(vi_gradient[0]) || mxIsNaN(vi_gradient[1])){
						#pragma omp critical
						{
							std::cout << "\n Nan in GRADIENT at vertex=" << vi;
						}
					}
				}

				if (grad_sq_norm < 1e-16)
				{
					if (!block_iter)
					{
						if (!model.return_search_dir)
							for (auto vi : free_vertex_block)
							{
								(*updated)[vi] = original[vi];

								if (mxIsNaN((*updated)[vi][0]) || mxIsNaN((*updated)[vi][1]))
								{
								    #pragma omp critical
									{
										std::cout << "\n Nan in Stationary-Block update in vertex" << vi;
									}
								}
							}
						else
							for (auto vi : free_vertex_block)
							{
								search_direction(2 * vi) = 0;
								search_direction(2 * vi + 1) = 0;
							}
					}
					break;
				}

				int solver_num = solverSpec.solver_num;
				if (ver_num == 1){
					solver_num = 0;
				}
				switch (solver_num) {
				case 0:
					GD_SearchDirectionInBlock(original, model, updated, solverSpec,
						gradient, search_direction, element_block,
						free_vertex_block, bnd_vertex_block);
					break;

				case 1:
					PN_SearchDirectionInBlock(original, model, updated, solverSpec,
						gradient, search_direction, element_block,
						free_vertex_block, bnd_vertex_block);
					break;

				default:
					mexErrMsgTxt("Block coordinate solver is not supported ");
					break;
				}
		}
		if (model.return_search_dir){
			return;
		}

		double step_time;
		double max_filter_time = 1.0;
		if (solverSpec.constant_step_size > 0){
			step_time = solverSpec.constant_step_size;
		} else {
			auto search_times = common::optimization::compute_min_step_to_singularities_inBlock(original,
														model.mesh_, search_direction, element_block, model);
			step_time = (solverSpec.is_flip_barrier) ?
				search_times[0] :
				std::min(search_times[1], max_filter_time);

			step_time = std::min(step_time, max_filter_time)*model.ls_interval;
		}

		if (mxIsNaN(step_time)){
			#pragma omp critical
			{
				std::cout << "\n step_time is NaN";
			}
		}
		double reduced_energy =
			common::optimization::ArmijoLineSearchEnhancedInBlock(
				original, step_time * search_direction, model, updated,
				gradient, fk, element_block, free_vertex_block);

		if (reduced_energy > fk) {
			for (auto vi : free_vertex_block) {
				(*updated)[vi] = original[vi];
			}
			return;
		}

		for (auto vi : free_vertex_block){
			original[vi] = (*updated)[vi];
		}
	}
}

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
	const std::vector<std::vector<int>>&  blocks_by_color)

{
	size_t color_num = blocks_by_color.size(),
		block_num = element_blocks.size();
	int max_iter_num = solverSpec.max_global_iterations;

	std::map<int, bool>	thread_calls;

	if (solverSpec.is_parallel && color_num) {
		std::cout << " Running in parallel\n";
	}
	else {
		std::cout << " Running sequentially";
		if (color_num)
		{
			std::cout << " with block coloring order";
		}
		std::cout << std::endl;
	}
	for (int i = 0; i < max_iter_num; i++) {
		std::cout << ((solverSpec.solver_num) ? "PN" : "GD") << " Iteration " << i; // << std::endl;

		if (color_num) {
			for (int ci = 0; ci < color_num; ci++) {
				int parallel_block_num = blocks_by_color[ci].size();
#pragma omp parallel for schedule(dynamic) if(solverSpec.is_parallel)
				for (int i = 0; i < parallel_block_num; i++) {
					int bi = blocks_by_color[ci][i];
					#pragma omp critical
					thread_calls[omp_get_thread_num()] = 1;

					UpdateSolutionInBlock(updated_copy, model, updated, solverSpec,
						gradient, search_direction,
						element_blocks[bi], free_vertex_blocks[bi], bnd_vertex_blocks[bi]);
				}
			}
		}
		else {
			for (int bi = 0; bi < block_num; bi++) {
				UpdateSolutionInBlock(updated_copy, model, updated, solverSpec,
					gradient, search_direction,
					element_blocks[bi], free_vertex_blocks[bi], bnd_vertex_blocks[bi]);
			}
		}

		if (common::optimization::IsNumericallyConverged(*original, *updated)) {
			//std::cout << "\n stop due to the convergence criteria";
			break;
		}

		if (!model.return_search_dir) {
			std::vector<Eigen::Vector2d>* temp = original;
			original = updated;
			updated = temp;
		}

	}
	std::cout << "\nThreads used :";
	for (auto ti : thread_calls){
		std::cout << ti.first << " ,";
	}
	std::cout << "\n";
}
