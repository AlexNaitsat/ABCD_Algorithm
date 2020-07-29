// Copyright @2020. All rights reserved.
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
#include "ABCD_optimization_core_3d.h"
#include <time.h>


size_t  BuiltConnectivityGraphFreeBlocks(int mesh_edges_num,
										int vert_num,
										SolverSpecification& solverSpec,
										Eigen::MatrixXd& source,
										const std::vector<Eigen::Vector3d>& target,
										Eigen::VectorXd& search_direction,
										const double* vv_mesh_edges,
										double K,
										std::vector<Eigen::Vector2i>& vv_graph_edges,
										std::vector<bool>& is_stationary_vert)
{
	vv_graph_edges.clear();
	int target_dim = target[0].size();
	std::vector<double> Luv(mesh_edges_num, 0);
	if (is_stationary_vert.size() != vert_num)
		is_stationary_vert.resize(vert_num, false);
	double Luv_max = -1;
	std::vector<size_t> proper_edges;

	bool is_loc_glob_blending = (solverSpec.K_hat < 1); //local-global blending flag
	proper_edges.reserve(mesh_edges_num);

	for (int i = 0; i < mesh_edges_num; i++) {
		size_t u = vv_mesh_edges[i], v = vv_mesh_edges[i + mesh_edges_num];
		double d_uv_norm = 0;
		Eigen::Vector3d d_u(target_dim), d_v(target_dim);
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

		if (is_u_stationary || is_v_stationary) {
			continue;
		}

		if (is_loc_glob_blending) { //local-global blending 
			proper_edges.push_back(i);

			auto d_uv = d_u - d_v; //Lipschitz coefficient
			double x_uv_norm = x_uv.norm();
			Luv[i] = (x_uv_norm > 1e-16) ? d_uv.norm()*y_uv.norm() / x_uv_norm : INFINITY;
			if (Luv[i] > Luv_max) {
				Luv_max = Luv[i];
			}
		}
		else {
			vv_graph_edges.push_back(Eigen::Vector2i(std::max(u, v), std::min(u, v)));
		}
	}

	if (is_loc_glob_blending) {
		for (auto i : proper_edges) {
			size_t u = vv_mesh_edges[i], v = vv_mesh_edges[i + mesh_edges_num];
			if ((Luv[i] < solverSpec.K_hat * Luv_max * 1.0001))
			{
				vv_graph_edges.push_back(Eigen::Vector2i(std::max(u, v), std::min(u, v)));
			}
		}
	}

	//adding  the stationary block
	int first_stationary_vert = -1;
	for (int v = 0; v < vert_num; v++) {
		if (is_stationary_vert[v])
		{
			if (first_stationary_vert == -1)
				first_stationary_vert = v;
			else
				vv_graph_edges.push_back(Eigen::Vector2i(v, first_stationary_vert));
		}
	}

	size_t unique_graph_edges_num = vv_graph_edges.size();
	return unique_graph_edges_num;
}


void GD_SearchDirectionInBlock(const std::vector<Eigen::Vector3d>& original,
								mesh_distortion::IsotropicSVDEnergyModel3D& model,
								SolverSpecification& solverSpec,
								Eigen::VectorXd& gradient,
								Eigen::VectorXd& search_direction,
								const std::vector<int>& element_block,
								const std::vector<int>& free_vertex_block,
								const std::vector<int>& bnd_vertex_block) 
{

	for (auto vi : free_vertex_block) {
		search_direction[3 * vi]     = -gradient[3 * vi];
		search_direction[3 * vi + 1] = -gradient[3 * vi + 1];
		search_direction[3 * vi + 2] = -gradient[3 * vi + 2];
	}
}

void PN_SearchDirectionInBlock(const std::vector<Eigen::Vector3d>& original,
								mesh_distortion::IsotropicSVDEnergyModel3D& model,
								SolverSpecification& solverSpec,
								Eigen::VectorXd& gradient,
								Eigen::VectorXd& search_direction,
								const std::vector<int>& element_block,
								const std::vector<int>& free_vertex_block,
								const std::vector<int>& bnd_vertex_block,
								common::solver::LinearSolver* m_solver)
{
	assert(m_solver != nullptr);
	std::vector<common::solver::eigen::EigenSolver::EigenEntry> entry_list;
	int ver_num = free_vertex_block.size();

	clock_t Hessian_computation_begin=clock();
	model.ComputeHessianNonzeroEntriesDirConstraintsInBlock(original, &entry_list,
		element_block, solverSpec);


	if (!solverSpec.use_pardiso_solver) {
		clock_t begin = clock();
		m_solver->SetPattern(entry_list, 3 * ver_num);
		if (!m_solver->IsPatternAnalyzed()) //single symbolic factorization for the same sparsity 
			m_solver->AnalyzePattern();
		clock_t middle = clock();
		m_solver->Factorize();
		clock_t end_factorize = clock();
		m_solver->SolveDirichletConstraints(-gradient, &search_direction, 
											free_vertex_block, bnd_vertex_block,3);
		clock_t end = clock();
	} else {
		#ifndef _USE_PARDISO
		#pragma omp critical 
				{
					std::cout << "\n **** To use Pardiso solver recompile mex with  _USE_PARDISO flag ***" << std::endl;
			    }
		#endif		

#ifdef _USE_PARDISO
		clock_t begin = clock();
		if (! m_solver->IsInitialized())
			m_solver->Init(-2);
		clock_t after_init = clock();
		if (! m_solver->IsPatternAnalyzed()) {
			m_solver->SetPattern(entry_list, 3 * ver_num);
			m_solver->AnalyzePattern();
		} else {
			m_solver->UpdateMatrixEntryValue(entry_list);
		}

		clock_t after_set_pattern = clock();
		clock_t middle = clock();
		m_solver->Factorize();
		clock_t end_factorize = clock();
		m_solver->SolveDirichletConstraints(-gradient, &search_direction,
			                                free_vertex_block, bnd_vertex_block,3);
		clock_t end = clock();

		#endif 
	}
}

void UpdateSolutionInBlock( std::vector<Eigen::Vector3d>& original,
							mesh_distortion::IsotropicSVDEnergyModel3D& model,
							std::vector<Eigen::Vector3d>* updated,
							SolverSpecification& solverSpec,
							Eigen::VectorXd& gradient,
							Eigen::VectorXd& search_direction,
							const std::vector<int>& element_block,
							const std::vector<int>& free_vertex_block,
							const std::vector<int>& bnd_vertex_block)
{

	if (!free_vertex_block.size() ||
		model.is_stationary_vertex[free_vertex_block.front()]) //skip stationary blocks
	{
		return;
	}

	int ver_num = free_vertex_block.size();
	for (auto vi : bnd_vertex_block) {
		search_direction(3 * vi)	= 0;
		search_direction(3 * vi + 1) = 0;
		search_direction(3 * vi + 2) = 0;
		(*updated)[vi] = original[vi];
		model.blockFreeVertIndex[vi] = -1;
	}

	for (size_t i = 0; i < ver_num; i++) {
		int v = free_vertex_block[i];
		model.blockFreeVertIndex[v] = i;
	}
	//initializing linear solver;
	common::solver::eigen::EigenSolver     local_eigen_solver;
#ifdef _USE_PARDISO
	common::solver::pardiso::PardisoSolver local_pardiso_solver;
#endif 

	common::solver::LinearSolver* m_solver_ptr = NULL;
	bool is_global_solver = (solverSpec.is_global || model.return_search_dir || (solverSpec.non_empty_block_num == 1) );
	if (!solverSpec.use_pardiso_solver) {
		m_solver_ptr = (is_global_solver) ?
			&model.global_eigen_solver : &local_eigen_solver;
	}
#ifdef _USE_PARDISO
	else {
		m_solver_ptr = (is_global_solver) ?
			&model.global_pardiso_solver : &local_pardiso_solver;
	};
#endif 

	for (int block_iter = 0; block_iter < solverSpec.max_block_iterations; block_iter++) {
		//std::cout << "\nblock_iter=" << block_iter;
		double fk = 0, grad_sq_norm;
		if (solverSpec.is_distortion_data_updated) {
			fk = model.prev_energy;
			solverSpec.is_distortion_data_updated = false;
		} else {
				fk = model.ComputeEnergyInBlock(original, element_block, solverSpec, true);
				double grad_sq_norm = 0;
				model.ComputeGradientInBlock(original, &gradient, element_block, 
											free_vertex_block, solverSpec);
				for (auto vi : free_vertex_block) {
					Eigen::Vector3d vi_gradient(gradient[3 * vi], gradient[3 * vi + 1], gradient[3 * vi + 2]);
					grad_sq_norm += vi_gradient.squaredNorm();

					if (mxIsNaN(vi_gradient[0]) || mxIsNaN(vi_gradient[1]) || mxIsNaN(vi_gradient[2])) {
						#pragma omp critical
						{
							std::cout << "\n Nan in GRADIENT at vertex=" << vi;
							mexErrMsgTxt("NAN value is found");
						}
					}
				}
				if (grad_sq_norm < 1e-16)
				{
					if (!block_iter)
					{
						if (!model.return_search_dir) {
							for (auto vi : free_vertex_block)
							{
								(*updated)[vi] = original[vi];

								if (mxIsNaN((*updated)[vi][0]) || mxIsNaN((*updated)[vi][1]) || mxIsNaN((*updated)[vi][2])) {
								#pragma omp critical
									{
										std::cout << "\n Nan in Stationary-Block update in vertex" << vi;
										mexErrMsgTxt("NAN value is found");
									}
								}
							}
						}else
							for (auto vi : free_vertex_block)
							{
								search_direction(3 * vi)     = 0;
								search_direction(3 * vi + 1) = 0;
								search_direction(3 * vi + 2) = 0;
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
					GD_SearchDirectionInBlock(original, model, solverSpec,
						gradient, search_direction, element_block,
						free_vertex_block, bnd_vertex_block);
					break;

				case 1:
					PN_SearchDirectionInBlock(original, model,  solverSpec,
						gradient, search_direction, element_block,
						free_vertex_block, bnd_vertex_block, m_solver_ptr);
					break;

				default:
					mexErrMsgTxt("This solver is not supported ");
					break;
				}
		}

		
		if (model.return_search_dir){
			return;
		}

		double step_time;
		double max_filter_time = 1.0;

		clock_t begin_ls_filtering = clock();
		Eigen::Vector3d search_times(-1.0, -1.0,-1.0);
		if (solverSpec.constant_step_size > 0) {
			step_time = solverSpec.constant_step_size;
		} else {
			 search_times = common::optimization::compute_min_step_to_singularities_3d_inBlock(original,
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
				mexErrMsgTxt("NAN value is found");
			}
		}
		clock_t begin_ls_armijo = clock();
		double reduced_energy =
			common::optimization::ArmijoLineSearchEnhancedInBlock3D(
				original, step_time * search_direction, model, updated,
				gradient, fk, element_block, free_vertex_block,solverSpec);

		clock_t end_ls_armijo = clock();

		if (reduced_energy > fk) {
			for (auto vi : free_vertex_block) {
				(*updated)[vi] = original[vi];
			}
			return;
		}
		if (block_iter + 1 < solverSpec.max_block_iterations //check for block convergance 
			&& common::optimization::IsNumericallyConverged3DInBlock(original, *updated, free_vertex_block)) {
			return; 
		}  else {
			for (auto vi : free_vertex_block) 	original[vi] = (*updated)[vi];
		}
	}
}

void OptimizeABCD(SolverSpecification& solverSpec,
	mesh_distortion::IsotropicSVDEnergyModel3D& model,
	Eigen::VectorXd& gradient,
	Eigen::VectorXd& search_direction,
	std::vector<Eigen::Vector3d>* original,
	std::vector<Eigen::Vector3d>*  updated,
	const std::vector<std::vector<int>>& element_blocks,
	const std::vector<std::vector<int>>& free_vertex_blocks,
	const std::vector<std::vector<int>>& bnd_vertex_blocks,
	const std::vector<std::vector<int>>&  blocks_by_color)

{
	size_t color_num = blocks_by_color.size(),
		block_num = element_blocks.size();
	int max_iter_num = solverSpec.max_global_iterations;

	std::map<int, bool>	thread_calls;

	if (color_num) {
			for (int ci = 0; ci < color_num; ci++) {
				int parallel_block_num = blocks_by_color[ci].size();
#pragma omp parallel for  if(solverSpec.is_parallel)
				for (int i = 0; i < parallel_block_num; i++) {
					int bi = blocks_by_color[ci][i];
					//std::cout << " \n bi=" << bi;
					#pragma omp critical
					thread_calls[omp_get_thread_num()] = 1;
					clock_t update_in_block_begin = clock();

					UpdateSolutionInBlock(*original, model, updated, solverSpec,
						gradient, search_direction, element_blocks[bi],
						free_vertex_blocks[bi], bnd_vertex_blocks[bi]);
				

				}
			}
		}
		else {
			for (int bi = 0; bi < block_num; bi++) {

				UpdateSolutionInBlock(*original, model, updated, solverSpec,
					gradient, search_direction,
					element_blocks[bi], free_vertex_blocks[bi], bnd_vertex_blocks[bi]);
			}
		}
}
