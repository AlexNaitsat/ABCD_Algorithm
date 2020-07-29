// Copyright @2019. All rights reserved.
// anaitsat@campus.technion.ac.il  (Alexander Naitsat)

#include "stdafx.h"
					
#include <iostream>
#include <vector>
#include <mex.h>
#include <Eigen/Dense>
#include "common/optimization/line_search_util.h"
#include "common/optimization/stop_check_util.h"
#include "common/solver/eigen/eigen_solver.h"
#ifdef _USE_PARDISO
	#include "common/solver/pardiso/pardiso_solver.h"
#endif 
#include "energy_model/distortion_kernel/arap_kernel_2d.h"
#include "energy_model/distortion_kernel/flip_penalty_kernel_2d.h"					 
//#include "energy_model/distortion_kernel/symmetric_dirichlet_kernel_2d.h"
#include "energy_model/distortion_kernel/sym_dirichlet_filtered_kernel_2d.h"
#include "energy_model/mesh_distortion/isotropic_svd_energy_model_2d.h"
#include <fstream>
#include <string>
#include "mat.h"

#include <omp.h>
#include <time.h>
#include <chrono>
#include "data_io/data_io_utils.h"
#include "common/util/graph_util.h"

using namespace data_io;
//#define  USE_PARDISO 1
#define  EXE_MEX_TEST_PROJECT 1

#include "ABCD_optimization_core_2d.h"

enum RunTime { InputUpload,SearchDirComputation, BlockProcessing, RestInitialization, Optimization, OutputSave };

void mexFunction(int num_of_lhs,
	mxArray* pointer_of_lhs[],
	int num_of_rhs,
	const mxArray* pointer_of_rhs[]) {

	const int Flip_penalty_num = 3;
	double runtime_seconds[7] = {0,0,0,0,0,0,0};

	double *position_rest, *position_deformed, *mesh, *num_of_vertex,
		*num_of_element, *kernel_type, *max_iteration,
		*solver_number, *energy_spec;

	const mxArray *element_blocks_pntr, *free_vertex_blocks_pntr,
		*bnd_vertex_blocks_pntr, *blocks_by_color_ptr = NULL;

	clock_t begin_input_upload = clock();

	position_rest = mxGetPr(pointer_of_rhs[0]);
	position_deformed = mxGetPr(pointer_of_rhs[1]);
	mesh = mxGetPr(pointer_of_rhs[2]);
	num_of_vertex = mxGetPr(pointer_of_rhs[3]);
	num_of_element = mxGetPr(pointer_of_rhs[4]);
	std::cout << "\n-------------- ABCD mex solver------";
	SolverSpecification solverSpec[2]; //fixer and optimizer specs.
	const mxArray* struct_pnt[2] = { pointer_of_rhs[5], pointer_of_rhs[6] };
	int source_dim = mxGetN(pointer_of_rhs[0]);
	
	double spd_thresh = 1e-6;
	//iterating over solver spec for fixer and optimizer 
	for (int pi = 0; pi < 2; pi++) {
		if (mxGetClassID(struct_pnt[pi]) != mxSTRUCT_CLASS)
		{
			mexErrMsgTxt("9th and 10th  parameters should be struct");
		}

		solverSpec[pi].invalid_penalty.resize(3, 0); //move to solver spec.
		solverSpec[pi].block_iteration_range.resize(3, 1);
		//solverSpec[pi].spd_thresh = 1e-6;

		FieldDataToArray<int>(struct_pnt[pi], "solver_num", &solverSpec[pi].solver_num, 1);
		FieldDataToArray<int>(struct_pnt[pi], "energy_num", &solverSpec[pi].energy_num, 1);
		FieldDataToArray<int>(struct_pnt[pi], "energy_num", &solverSpec[pi].energy_num, 1);
		FieldDataToArray<int>(struct_pnt[pi], "cycle_num", &solverSpec[pi].cycle_num, 1);
		FieldDataToArray<bool>(struct_pnt[pi], "is_global", &solverSpec[pi].is_global, 1);
		FieldDataToArray<bool>(struct_pnt[pi], "use_pardiso", &solverSpec[pi].use_pardiso_solver, 1);

		FieldDataToArray<bool>(struct_pnt[pi], "single_fixed_block", &solverSpec[pi].single_fixed_block, 1);
		FieldDataToArray<double>(struct_pnt[pi], "invalid_penalty", solverSpec[pi].invalid_penalty, 3);
		FieldDataToArray<double>(struct_pnt[pi], "block_iteration_range", solverSpec[pi].block_iteration_range, 3);
		solverSpec[pi].max_block_iterations = solverSpec[pi].block_iteration_range[0];//initalizing block iteration 

		FieldDataToArray<int>(struct_pnt[pi], "max_global_iterations", &solverSpec[pi].max_global_iterations, 1);
		FieldDataToArray<bool>(struct_pnt[pi], "is_signed_svd", &solverSpec[pi].is_signed_svd, 1);
		FieldDataToArray<bool>(struct_pnt[pi], "is_parallel", &solverSpec[pi].is_parallel, 1);
		FieldDataToArray<bool>(struct_pnt[pi], "is_parallel_grad", &solverSpec[pi].is_parallel_grad, 1);
		FieldDataToArray<bool>(struct_pnt[pi], "is_parallel_energy", &solverSpec[pi].is_parallel_energy, 1);
		FieldDataToArray<bool>(struct_pnt[pi], "is_parallel_hessian", &solverSpec[pi].is_parallel_hessian, 1);

		FieldDataToArray<bool>(struct_pnt[pi], "is_flip_barrier", &solverSpec[pi].is_flip_barrier, 1);
		FieldDataToArray<bool>(struct_pnt[pi], "report_data", &solverSpec[pi].report_data, 1);

		FieldDataToArray<double>(struct_pnt[pi], "constant_step_size", &solverSpec[pi].constant_step_size, 1);
		FieldDataToArray<int>(struct_pnt[pi], "source_dim", &source_dim, 1);


		FieldDataToArray<double>(struct_pnt[pi], "zero_grad_eps", &solverSpec[pi].zero_grad_eps, 1);
		FieldDataToArray<bool>(struct_pnt[pi], "is_flip_barrier", &solverSpec[pi].is_flip_barrier, 1);
		FieldDataToArray<double>(struct_pnt[pi], "K_hat", &solverSpec[pi].K_hat, 1);
		FieldDataToArray<double>(struct_pnt[pi], "tolerance", &solverSpec[pi].energy_tolerance, 1);


		//Local-global blending params
		FieldDataToArray<int>(struct_pnt[pi], "K_hat_size", &solverSpec[pi].K_hat_size, 1);
		std::vector<double> K_hat_list(solverSpec[pi].K_hat_size, 2.0);
		FieldDataToArray<double>(struct_pnt[pi], "K_hat_list", K_hat_list, solverSpec[pi].K_hat_size);

		for (auto K : K_hat_list)
			solverSpec[pi].K_hat_queue.push(K);

	}

	int tri_num = num_of_element[0];
	int ver_num = num_of_vertex[0];
	
	int color_num = 0;
	Eigen::MatrixXd rest(ver_num,source_dim);
	std::vector<Eigen::Vector3i> triangle(tri_num);
	std::vector<Eigen::Vector2d> deformed(ver_num);
	
	for (size_t i = 0; i < tri_num; i++) {
		triangle[i][0] = mesh[i + 0 * tri_num] - 1;
		triangle[i][1] = mesh[i + 1 * tri_num] - 1;
		triangle[i][2] = mesh[i + 2 * tri_num] - 1;
	}

	for (size_t i = 0; i < ver_num; i++) {
		rest(i,0) = position_rest[i + 0 * ver_num];
		rest(i,1) = position_rest[i + 1 * ver_num];
		
		if (source_dim==3){
			rest(i, 2) = position_rest[i + 2 * ver_num];
		}

		deformed[i][0] = position_deformed[i + 0 * ver_num];
		deformed[i][1] = position_deformed[i + 1 * ver_num];
	}

	runtime_seconds[RunTime::InputUpload] = double(clock() - begin_input_upload) / CLOCKS_PER_SEC;

	distortion_kernel::DistortionKernel2D* kernel[2] = { NULL,NULL};

	bool is_inversion_free_stage_global = true; //BCD or global optimization after fixing inversions 
	for (int pi = 0; pi < 2; pi++) {
		switch (solverSpec[pi].energy_num) {
		case 0:
			std::cout << "\nmeasure#"<<pi <<" ARAP energy";
			kernel[pi] = new distortion_kernel::ARAPKernel2D();
			break;
		case 2:
			std::cout << "\nmeasure#" << pi << " Modified Symmetric Dirichlet energy";
			kernel[pi] = new distortion_kernel::SymDirichletFilteredKernel2D();
			kernel[pi]->EnableFlipFilter(true);
			break;
		case 3:
			kernel[pi] = new distortion_kernel::FlipPenaltyKernel2D(solverSpec[pi].invalid_penalty);
			std::cout << "\nmeasure#" << pi << " Flip  Penalty energy, invalid penalties=" << solverSpec[pi].invalid_penalty;
			break;
		default:
			std::cout << "\n Input energy number " << solverSpec[pi].energy_num << " is not supported ";
			mexErrMsgTxt("\n Unsupported energy");
			return;
			break;

		}
		std::cout << ", Linear solver: " << ((solverSpec[pi].use_pardiso_solver) ? "Pardiso" : "Eigen")
			<< ", parallel flags: " << solverSpec[pi].is_parallel << ","
			<< solverSpec[pi].is_parallel_energy << ","
			<< solverSpec[pi].is_parallel_grad << ","
			<< solverSpec[pi].is_parallel_hessian<<std::endl;
		
		#ifndef _USE_PARDISO
			if (solverSpec[pi].use_pardiso_solver) 
				mexErrMsgTxt("\nPardiso code is not linked, set _USE_PARDISO macro  in linear_solver.h and recompile");
		#endif
	}


	//----common model initialization ----
	clock_t begin_init = clock(), BCD_invalids_fixed = 0;
	mesh_distortion::IsotropicSVDEnergyModel2D model;
	model.SetRestMesh(rest, triangle);
	model.SetDirichletConstraints(ver_num, -1);
	

	//---- initializing vertex buffers 
	std::vector<Eigen::Vector2d> buffer = deformed, updated_copy = deformed;
	std::vector<Eigen::Vector2d>* original = &deformed;
	std::vector<Eigen::Vector2d>* updated = &buffer;

	Eigen::VectorXd search_direction(2 * ver_num),
		gradient(2 * ver_num);
	search_direction.setZero();
	gradient.setZero();
	//---------------------------------
	int iter_num =(int) (*mxGetPr(pointer_of_rhs[7]));


	//-------------- initializaing block relatated data ----------------------------------
	int mesh_struct_rhs_index = (num_of_rhs >= 10)? 10+2 : 6+2; //iter num and another spec are two additional  inputs
	const mxArray* mesh_struct_pnt = pointer_of_rhs[mesh_struct_rhs_index];
	double  *vv_mesh_edges;
	int mesh_edges_num = 0;

	std::vector<std::vector<int>> vert_simplices, vert_neighbors,
								all_elements, free_vertices,
								fixed_vertices;

	int vertex_num;
	//put it before the alternating loop.
	std::vector<std::vector<int>>	single_block_color(1, std::vector<int>{0});
	FieldDataToPointer(mesh_struct_pnt, "vv_mesh_edges", &vv_mesh_edges, &mesh_edges_num);
	FieldCellArrayToVectorArray<int>(mesh_struct_pnt, "vert_simplices", vert_simplices);
	FieldCellArrayToVectorArray<int>(mesh_struct_pnt, "vert_neighbors", vert_neighbors);
	FieldCellArrayToVectorArray<int>(mesh_struct_pnt, "all_elements", all_elements);
	FieldCellArrayToVectorArray<int>(mesh_struct_pnt, "free_vertices", free_vertices);
	FieldCellArrayToVectorArray<int>(mesh_struct_pnt, "fixed_vertices", fixed_vertices);

	FieldDataToArray<int>(mesh_struct_pnt, "vert_num", &vertex_num, 1);
	FieldDataToArray<int>(mesh_struct_pnt, "tri_num", &tri_num, 1);
	FieldDataToArray<int>(mesh_struct_pnt, "vert_num", &vertex_num, 1);
	
	std::vector<bool> is_fixed_vert(vertex_num);
	FieldDataToArray<bool>(mesh_struct_pnt, "is_fixed_vert", is_fixed_vert, vertex_num);

	std::map<int, bool>	thread_calls;
	std::vector<int>    invalids_per_iter, energy_num_per_iter;
	std::vector<double> time_per_iter, energy_per_iter;
	//----------------------------------------------------------------------------------


	//Alternating loop between optimizer and fixer 
	bool is_disotrtion_optimized[2] = { false, false },
		preserve_prev_block_data = false;
	int total_iter = 0;
	
	std::vector<std::vector<int>> blocks_by_color, blocks_by_color_sorted,
		free_vertex_blocks, element_blocks,
		bnd_vertex_blocks;
	clock_t BCD_optimization_begin = clock();
	bool is_converged = false;
	while (total_iter < iter_num && !is_converged) {
		for (int pi = 0; pi < 2; pi++) {
			if (is_disotrtion_optimized[pi])
				continue;
			for (int cycle_iter = 0; (cycle_iter < solverSpec[pi].cycle_num) && (total_iter < iter_num) ; cycle_iter++) {
				if (!preserve_prev_block_data) {
						blocks_by_color.clear();
						blocks_by_color_sorted.clear();
						free_vertex_blocks.clear();
						element_blocks.clear();
						bnd_vertex_blocks.clear();
				}

				//----switching between fixer and optimizer models---- 
				FieldDataToArray<double>(struct_pnt[pi], "ls_interval", &model.ls_interval, 1);
				FieldDataToArray<double>(struct_pnt[pi], "ls_alpha", &model.ls_alpha, 1);
				FieldDataToArray<double>(struct_pnt[pi], "ls_beta", &model.ls_beta, 1);
				FieldDataToArray<bool>(struct_pnt[pi], "is_flip_barrier", &model.is_flip_barrier, 1);
				FieldDataToArray<int>(struct_pnt[pi], "ls_max_iter", &model.ls_max_iter, 1);
				FieldDataToArray<bool>(struct_pnt[pi], "return_search_dir", &model.return_search_dir, 1);
				model.SetDistortionKernel(kernel[pi]);
				model.SetEnforcingSPD(true, spd_thresh);
				model.SetSignedSVD(solverSpec[pi].is_signed_svd);
				//----------------------------------------------------
				int block_num = 1;
				if (solverSpec[pi].is_global) {
					if (!preserve_prev_block_data) {
						free_vertex_blocks = free_vertices;
						bnd_vertex_blocks = fixed_vertices;
						element_blocks = all_elements;
						blocks_by_color = single_block_color;
						blocks_by_color_sorted = single_block_color;
					}
				}
				else {
					solverSpec[pi].non_empty_block_num = 1;//for descent direction compution 

					SolverSpecification search_dirSpec = solverSpec[pi];
					search_dirSpec.is_parallel = false;

					model.return_search_dir = true;
					model.invalid_element_num = 0;
					size_t search_direction_begin = clock();

					solverSpec[pi].non_empty_block_num = 1;
					model.is_stationary_vertex.assign(vertex_num, false);
					OptimizeABCD(search_dirSpec, model,
						gradient, search_direction,
						original, updated, updated_copy,
						all_elements, free_vertices, fixed_vertices,
						single_block_color
					);

					
					model.return_search_dir = false;
					
					size_t search_direction_end = clock();
					runtime_seconds[RunTime::SearchDirComputation] = double(search_direction_end - search_direction_begin) / CLOCKS_PER_SEC;

					if (is_inversion_free_stage_global && !model.invalid_element_num) {
						BCD_invalids_fixed = clock();
						std::cout << "\n All invalid simplices are fixed, running only the optimizer";
						std::cout << "\n      #measure=" << pi << ", energy_num=" << solverSpec[pi].energy_num << ", Energy=" << model.prev_energy;

						for (int j = 0; j < 2; j++) {
							solverSpec[j].is_global = true; //bcd  to global switch
							if (solverSpec[j].energy_num == Flip_penalty_num) {
								is_disotrtion_optimized[j] = true;
							}

						}
						solverSpec[pi].non_empty_block_num = 1;

						
						free_vertex_blocks = free_vertices; //initialize single block data 
						bnd_vertex_blocks = fixed_vertices;
						element_blocks = all_elements;
						blocks_by_color = single_block_color;
						blocks_by_color_sorted = single_block_color;
						preserve_prev_block_data = true;//use the same block data
					}
					if (is_disotrtion_optimized[pi]) {
						std::cout << " (optimized)";
						break; //switch to the next distortion 
					}

					if (! solverSpec[pi].is_global) //block partitioning part
					{
						clock_t block_processing_begin = clock();
						std::vector<Eigen::Vector2i> vv_graph_edges_vec;
						double K = 1e16;					
						if (solverSpec[pi].K_hat_size > 0) { //using input partitioning thesholds  
							if (solverSpec[pi].K_hat_queue.empty()) {
								solverSpec[pi].K_hat = 1.0;
							}
							else {
								solverSpec[pi].K_hat = solverSpec[pi].K_hat_queue.front();
								solverSpec[pi].K_hat_queue.pop();
							}
						}

						int graph_edges_num = 0;
							graph_edges_num = BuiltConnectivityGraphFreeBlocks(
								mesh_edges_num, vertex_num,
								solverSpec[pi], rest,
								*original,
								search_direction,
								vv_mesh_edges,
								K, 
								vv_graph_edges_vec,
								model.is_stationary_vertex
							);

							double* vv_graph_edges = new double[2 * graph_edges_num];
						for (int i = 0; i < vv_graph_edges_vec.size(); i++) {
							vv_graph_edges[i] = vv_graph_edges_vec[i](0);
							vv_graph_edges[i + graph_edges_num] = vv_graph_edges_vec[i](1);
						}

						std::vector<std::vector<int> >free_vertex_blocks_vec;
						std::vector<std::set<int> >element_blocks_set, bnd_vertex_blocks_set;

						mxArray *vertex_block_indexArr = mxCreateDoubleMatrix(1, vertex_num, mxREAL);
						double* vertex_block_index_ptr = mxGetPr(vertex_block_indexArr);

						size_t color_num1 =
							util::PartitionToBlocks(vertex_num, tri_num, mesh_edges_num, graph_edges_num,
								vert_simplices, vert_neighbors,
								vv_mesh_edges, vv_graph_edges,
								is_fixed_vert,
								element_blocks_set, free_vertex_blocks, bnd_vertex_blocks_set,
								vertex_block_index_ptr,
								blocks_by_color,
								solverSpec[pi].is_parallel
							);
						block_num = element_blocks_set.size();

						if (block_num != element_blocks_set.size()) {
							std::cout << "\n block_num differes from size of element_blocks_set!";
						}

						element_blocks.resize(block_num);
						bnd_vertex_blocks.resize(block_num);

						for (int b = 0; b < block_num; b++) {
							for (auto el : element_blocks_set[b]) {
								element_blocks[b].push_back(el);
							}
							for (auto bv : bnd_vertex_blocks_set[b]) {
								bnd_vertex_blocks[b].push_back(bv);
							}
						}

						std::vector<double> block_energy(block_num, 0), color_energy(color_num1, 0);
						for (int b = 0; b < block_num; b++) {
							for (auto t : element_blocks[b]) {
								block_energy[b] += model.element_distortion[t] * model.volume_[t];
							}
						}
						auto comp = [&](int i, int j) {
							return (color_energy[i] > color_energy[j]) || (color_energy[i] == color_energy[j] && i < j);
						};
						auto color_indices_sorted = std::set<int, decltype(comp)>(comp);

						std::set<double, std::greater<double>> color_energy_sorted;
						for (int c = 0; c < color_num1; c++) {
							for (auto b : blocks_by_color[c]) {
								color_energy[c] += block_energy[b];
							}
							color_indices_sorted.insert(c);
							color_energy_sorted.insert(color_energy[c]);
						}

						for (auto s_sorted : color_indices_sorted) {
							blocks_by_color_sorted.push_back(blocks_by_color[s_sorted]);
						}

						solverSpec[pi].non_empty_block_num = 0;
						for (const auto& b : free_vertex_blocks) {
							if (b.size() > 0) {
								solverSpec[pi].non_empty_block_num++;
							}
						}

						clock_t block_processing_end = clock();
						runtime_seconds[RunTime::BlockProcessing] = double(block_processing_end - block_processing_begin) / CLOCKS_PER_SEC;
					}
					solverSpec[pi].is_distortion_data_updated = (solverSpec[pi].non_empty_block_num == 1);
				}
				double current_timing = double(clock() - BCD_optimization_begin) / CLOCKS_PER_SEC;

				std::cout << "\n-----Iteration " << total_iter << ": " << ((solverSpec[pi].solver_num) ? "PN" : "GD")
					<< ", energy_num= " << solverSpec[pi].energy_num << ", Energy= " << model.prev_energy
					<< ", number of blocks= " << block_num << ", block iterations= "
					<< floor(solverSpec[pi].max_block_iterations) << ", #invalids= " << model.invalid_element_num
					<< ", time= " << current_timing <<
					", energy_diff= " << model.energy_difference << std::endl;

				model.return_search_dir = false;


				model.invalid_element_num = 0;

		
				OptimizeABCD(solverSpec[pi], model,
					gradient, search_direction,
					original, updated, updated_copy,
					element_blocks, free_vertex_blocks, bnd_vertex_blocks,
					blocks_by_color_sorted);

				total_iter++;
				solverSpec[pi].update_block_iteration();
				
				clock_t BCD_optimization_end = clock();
				runtime_seconds[RunTime::Optimization] = double(BCD_optimization_end - BCD_optimization_begin) / CLOCKS_PER_SEC;

				is_converged = (is_disotrtion_optimized[0] && (pi == 1) &&
					(abs(model.energy_difference) / (abs(model.prev_energy) + 1e-12) < solverSpec[pi].energy_tolerance)
					&& (abs(model.prev_energy) / model.total_volume < 1e7));

				if (is_converged) {
					std::cout << " Stop due to the energy convergence, tolerance= "
						<< solverSpec[pi].energy_tolerance << std::endl;
					break;
				}

			}
		}
	}
	
	//-------Reporting and output part of optimizer distortion--------------
	int pi = 1;//optimizer
	clock_t begin_output_save = clock();

		double* output;
		pointer_of_lhs[0] = mxCreateDoubleMatrix(deformed.size() * 2, 1, mxREAL);
		output = mxGetPr(pointer_of_lhs[0]);

		for (size_t i = 0; i < deformed.size(); i++) {
			output[2 * i + 0] = (*original)[i][0];
			output[2 * i + 1] = (*original)[i][1];
		}


	runtime_seconds[RunTime::OutputSave] = double(clock() - begin_output_save) / CLOCKS_PER_SEC;
	
	double total_runtime_seconds = 0;
	std::cout << "\n--Total runtime= " << double(clock() - begin_init) / CLOCKS_PER_SEC;

	if (BCD_invalids_fixed)
		std::cout << "\n--Total time for fixing all invalids= " << double(BCD_invalids_fixed - begin_init) / CLOCKS_PER_SEC;

	std::cout << "\n";
	return;
}
