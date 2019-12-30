// Copyright @2019. All rights reserved.
// Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)
#include "stdafx.h"
					
#include <iostream>
#include <vector>
#include <mex.h>
#include <Eigen/Dense>
#include "common/optimization/line_search_util.h"
#include "common/optimization/stop_check_util.h"
#include "common/solver/eigen/eigen_solver.h"
#include "common/solver/pardiso/pardiso_solver.h"
#include "energy_model/distortion_kernel/arap_kernel_2d.h"
#include "energy_model/distortion_kernel/flip_penalty_kernel_2d.h"					 
#include "energy_model/distortion_kernel/symmetric_dirichlet_kernel_2d.h"
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

#include "common/solver/pardiso/pardiso_solver.h"
using namespace data_io;
#define  USE_PARDISO 1
#define  EXE_MEX_TEST_PROJECT 1

#include "ABCD_optimization_core.h"

enum RunTime { InputUpload,SearchDirComputation, BlockProcessing, RestInitialization, Optimization, OutputSave };

void mexFunction(int num_of_lhs,
	mxArray* pointer_of_lhs[],
	int num_of_rhs,
	const mxArray* pointer_of_rhs[]) {

	double runtime_seconds[7] = {0,0,0,0,0,0,0};

	double *position_rest, *position_deformed, *mesh, *num_of_vertex,
		*num_of_element, *kernel_type, *spd_threshold, *max_iteration,
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
	SolverSpecification solverSpec;
	const mxArray* struct_pnt = pointer_of_rhs[5];
	if (mxGetClassID(struct_pnt) != mxSTRUCT_CLASS)
	{
		mexErrMsgTxt("9th parameter should be struct");
	}

	std::vector <double> invalid_penalty(3, 0);
	double spd_thresh = 1e-6;

	FieldDataToArray<int>(struct_pnt, "solver_num", &solverSpec.solver_num, 1);
	FieldDataToArray<int>(struct_pnt, "energy_num", &solverSpec.energy_num, 1);
	FieldDataToArray<int>(struct_pnt, "energy_num", &solverSpec.energy_num, 1);
	FieldDataToArray<bool>(struct_pnt, "is_global",  &solverSpec.is_global,  1);
	FieldDataToArray<bool>(struct_pnt, "use_pardiso", &solverSpec.use_pardiso_solver, 1);
	FieldDataToArray<bool>(struct_pnt, "single_fixed_block", &solverSpec.single_fixed_block, 1);

	FieldDataToArray<double>(struct_pnt, "invalid_penalty", invalid_penalty, 3);
	FieldDataToArray<double>(struct_pnt, "invalid_penalty", invalid_penalty, 3);
	FieldDataToArray<double>(struct_pnt, "max_block_iterations", &solverSpec.max_block_iterations, 1);
	FieldDataToArray<int>(struct_pnt, "max_global_iterations", &solverSpec.max_global_iterations, 1);
	FieldDataToArray<bool>(struct_pnt, "is_signed_svd", &solverSpec.is_signed_svd, 1);
	FieldDataToArray<bool>(struct_pnt, "is_parallel", &solverSpec.is_parallel, 1);
	FieldDataToArray<bool>(struct_pnt, "is_flip_barrier", &solverSpec.is_flip_barrier, 1);
	FieldDataToArray<bool>(struct_pnt, "report_data", &solverSpec.report_data, 1);
	
	FieldDataToArray<double>(struct_pnt, "constant_step_size", &solverSpec.constant_step_size, 1);
	int source_dim = 2;
	FieldDataToArray<int>(struct_pnt, "source_dim",&source_dim ,1);

	int tri_num = num_of_element[0];
	int ver_num = num_of_vertex[0];
	int kernel_t = solverSpec.energy_num;
	
	int color_num = 0;
	int max_iter_num = solverSpec.max_global_iterations;

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
	//std::cout << "\n-Input upload time="   << runtime_seconds[RunTime::InputUpload];

#ifdef _TEST_PARSISO
		std::cout << "\n PARDISO: Testing pardiso_main.cc\n";
		std::vector<common::solver::pardiso::PardisoSolver::MatrixEntry> entry_list;
	
		entry_list.emplace_back(0, 0, 0.1516);
		entry_list.emplace_back(0, 0, 0.1516);
		entry_list.emplace_back(0, 2, 0.3222);
		entry_list.emplace_back(2, 0, 0.3222);
		entry_list.emplace_back(1, 1, 0.0068);
		entry_list.emplace_back(3, 3, 0.4694);
		entry_list.emplace_back(2, 3, 0.4466);
		entry_list.emplace_back(3, 2, 0.4466);
		entry_list.emplace_back(2, 2, 0.2339);
	
		std::vector<double> rhs(4);
		rhs[0] = 0.2974;
		rhs[1] = 0.4137;
		rhs[2] = 0.0119;
		rhs[3] = 0.8255;
	
		std::vector<double> res(4);
	
		common::solver::pardiso::PardisoSolver m_solver;
	
		m_solver.Init(11);
		m_solver.SetPattern(entry_list, 4);
		m_solver.AnalyzePattern();
		m_solver.Factorize();
		m_solver.Solve(rhs, &res);
	
		for (int i = 0; i < res.size(); i++)
		{
			std::cout << res[i] << std::endl;
		}
		std::cout << std::endl;
	
		entry_list.clear();
		entry_list.emplace_back(0, 0, 3.1516);
		entry_list.emplace_back(0, 0, 3.1516);
		entry_list.emplace_back(0, 2, 1.3222);
		entry_list.emplace_back(2, 0, 1.3222);
		entry_list.emplace_back(1, 1, 2.0068);
		entry_list.emplace_back(3, 3, 4.4694);
		entry_list.emplace_back(2, 3, 3.4466);
		entry_list.emplace_back(3, 2, 3.4466);
		entry_list.emplace_back(2, 2, 1.2339);
	
		m_solver.UpdateMatrixEntryValue(entry_list);
		m_solver.Factorize();
		m_solver.Solve(rhs, &res);
	
		for (int i = 0; i < res.size(); i++)
		{
			std::cout << res[i] << std::endl;
		}
		std::cout << std::endl;
	
		m_solver.FreeSolver();
#endif

	distortion_kernel::DistortionKernel2D* kernel = NULL;

	switch (solverSpec.energy_num) {
	case 0:
		std::cout << "\n ARAP energy";
		kernel = new distortion_kernel::ARAPKernel2D();
		break;
	case 1:
		std::cout << "\n Symmetric Dirichlet energy";
		kernel = new distortion_kernel::SymmetricDirichletKernel2D();
		break;
	case 2:
		std::cout << "\n Filtered Symmetric Dirichlet energy";
		kernel = new distortion_kernel::SymDirichletFilteredKernel2D();
		kernel->EnableFlipFilter(true);
		break;
	 case 3:
		kernel = new distortion_kernel::FlipPenaltyKernel2D(invalid_penalty);
		std::cout << "\n Flip  Penalty energy, invalid penalties=" << invalid_penalty;
		break;
	 default:
		 std::cout << "\n Input energy number "<< solverSpec.energy_num<< " is not supported ";
		 mexErrMsgTxt("\n Unsupported energy");
		 return;
		break;

	}

	clock_t begin_init = clock();
	mesh_distortion::IsotropicSVDEnergyModel2D model;
	model.SetRestMesh(rest, triangle);
	model.SetDistortionKernel(kernel);
	model.SetEnforcingSPD(true, spd_thresh);
	model.SetSignedSVD(solverSpec.is_signed_svd);
	clock_t end_init = clock();
	runtime_seconds[RunTime::RestInitialization] = double(end_init - begin_init) / CLOCKS_PER_SEC;
	//std::cout << "\nTotal rest mesh initalization time=" << runtime_seconds[RunTime::RestInitialization];

	FieldDataToArray<double>(struct_pnt, "ls_interval", &model.ls_interval, 1);
	FieldDataToArray<double>(struct_pnt, "ls_alpha",    &model.ls_alpha, 1);
	FieldDataToArray<double>(struct_pnt, "ls_beta", &model.ls_beta, 1);
	FieldDataToArray<double>(struct_pnt, "zero_grad_eps", &solverSpec.zero_grad_eps,1);
	FieldDataToArray<bool>(struct_pnt, "is_flip_barrier",&solverSpec.is_flip_barrier, 1);
	FieldDataToArray<bool>(struct_pnt, "is_flip_barrier", &model.is_flip_barrier, 1);
	FieldDataToArray<int>(struct_pnt, "ls_max_iter", &model.ls_max_iter, 1);
	FieldDataToArray<double>(struct_pnt, "K_hat", &solverSpec.K_hat, 1);

	FieldDataToArray<bool>(struct_pnt, "return_search_dir", &model.return_search_dir, 1);

	std::vector<Eigen::Vector2d> buffer=deformed, updated_copy = deformed;

	std::vector<Eigen::Vector2d>* original = &deformed; 
	std::vector<Eigen::Vector2d>* updated = &buffer;

	Eigen::VectorXd search_direction(2 * ver_num),
		gradient(2 * ver_num);
	search_direction.setZero();
	gradient.setZero();

	int mesh_struct_rhs_index = (num_of_rhs >= 10)? 10 : 6;
	const mxArray* mesh_struct_pnt = pointer_of_rhs[mesh_struct_rhs_index];
	double  *vv_mesh_edges;
	int mesh_edges_num = 0;

	std::vector<std::vector<int>> vert_simplices, vert_neighbors,
								all_elements, free_vertices,
								fixed_vertices;

	int vertex_num;
	
	std::vector<std::vector<int>> blocks_by_color, blocks_by_color_sorted,
		free_vertex_blocks, element_blocks,
		bnd_vertex_blocks,
		single_block_color(1, std::vector<int>{0});

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

	std::map<int,bool>	thread_calls;
	model.SetDirichletConstraints(ver_num,-1);

	if (solverSpec.is_global) {
		free_vertex_blocks = free_vertices;
		bnd_vertex_blocks = fixed_vertices;
		element_blocks = all_elements;
		blocks_by_color = single_block_color;
		blocks_by_color_sorted = single_block_color;
		std::cout << " Global ";
	} else {
		size_t non_empty_block_num = 0;

		SolverSpecification search_dirSpec = solverSpec;
		search_dirSpec.is_parallel = false;

		model.return_search_dir = true;
		size_t serach_direction_beign = clock();
		OptimizeABCD(search_dirSpec, model,
			gradient, search_direction,
			original, updated, updated_copy,
			all_elements, free_vertices, fixed_vertices,
			single_block_color
		);
		model.return_search_dir = false;
		size_t serach_direction_end = clock();
		runtime_seconds[RunTime::SearchDirComputation] = double(end_init - begin_init) / CLOCKS_PER_SEC;
		std::cout << "\n-Descent direction computation  time=" << runtime_seconds[RunTime::SearchDirComputation];

		clock_t block_processing_begin = clock();
		std::vector<Eigen::Vector2i> vv_graph_edges_vec;
		double K = 1e16;

		int graph_edges_num = 0;
		if (solverSpec.single_fixed_block) {
			graph_edges_num = BuiltConnectivityGraphSingleStationaryBlock(
				mesh_edges_num, vertex_num,
				model, solverSpec, rest,
				*original,
				search_direction,
				vv_mesh_edges,
				K,
				vv_graph_edges_vec);
		} else {
			graph_edges_num = BuiltConnectivityGraph(
				mesh_edges_num, vertex_num,
				model, solverSpec, rest,
				*original,
				search_direction,
				vv_mesh_edges,
				K,
				vv_graph_edges_vec);
		}

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
				solverSpec.is_parallel
			);
		int block_num = element_blocks_set.size();
		std::cout << "\nnumber of blocks =" << block_num << ","
			<< solverSpec.max_block_iterations << " block iterations";

		if (block_num != element_blocks_set.size()) {
			std::cout << "\n block_num differes from size of element_blocks_set";
		}

		element_blocks.resize(block_num);
		bnd_vertex_blocks.resize(block_num);

		for (int b = 0; b < block_num; b++) {
			for (auto el : element_blocks_set[b]){
				element_blocks[b].push_back(el);
			}
			for (auto bv : bnd_vertex_blocks_set[b]){
				bnd_vertex_blocks[b].push_back(bv);
			}
		}

		std::vector<double> block_energy(block_num, 0), color_energy(color_num1, 0);
		for (int b = 0; b < block_num; b++) {
			for (auto t : element_blocks[b]){
				block_energy[b] += model.element_distortion[t] * model.volume_[t];
			}
		}
		auto comp = [&](int i, int j) {
			return (color_energy[i] > color_energy[j]) || (color_energy[i] == color_energy[j] && i < j);
		};
		auto color_indices_sorted = std::set<int, decltype(comp)>(comp);

		std::set<double, std::greater<double>> color_energy_sorted;
		for (int c = 0; c < color_num1; c++) {
			for (auto b : blocks_by_color[c]){
				color_energy[c] += block_energy[b];
			}
			color_indices_sorted.insert(c);
			color_energy_sorted.insert(color_energy[c]);
		}

		for (auto s_sorted : color_indices_sorted){
			blocks_by_color_sorted.push_back(blocks_by_color[s_sorted]);
		}
		solverSpec.non_empty_block_num = 0;
		for (const auto& b : free_vertex_blocks) {
			if (b.size() > 0){
				solverSpec.non_empty_block_num++;
			}
		}
		solverSpec.is_distortion_data_updated = (solverSpec.non_empty_block_num == 1);
		
		clock_t block_processing_end = clock();
		runtime_seconds[RunTime::BlockProcessing] = double(block_processing_end - block_processing_begin) / CLOCKS_PER_SEC;
		std::cout << "\n-Total block processing  time=" << runtime_seconds[RunTime::BlockProcessing];
	}
	model.return_search_dir = false;

	clock_t BCD_optimization_begin = clock();
	OptimizeABCD(solverSpec, model,
				 gradient, search_direction,
				 original, updated, updated_copy,
				 element_blocks, free_vertex_blocks, bnd_vertex_blocks,
				 blocks_by_color_sorted);
	clock_t BCD_optimization_end = clock();
	runtime_seconds[RunTime::Optimization] = double(BCD_optimization_end - BCD_optimization_begin) / CLOCKS_PER_SEC;
	std::cout << "\n-Total optimization  time=" << runtime_seconds[RunTime::Optimization];

	clock_t begin_output_save = clock();

	if (!model.return_search_dir) {
		double* output;
		pointer_of_lhs[0] = mxCreateDoubleMatrix(deformed.size() * 2, 1, mxREAL);
		output = mxGetPr(pointer_of_lhs[0]);

		for (size_t i = 0; i < deformed.size(); i++) {
			output[2 * i + 0] = (*original)[i][0];
			output[2 * i + 1] = (*original)[i][1];
		}
	} else {
		pointer_of_lhs[0] = mxCreateDoubleMatrix(1, 0, mxREAL);
	}
	if (num_of_lhs >= 6) {
		const int simplex_dim = 2;
		//std::cout << "\n Returns search direction";
		double *output_grad, *output_search_dir, *output_sing_vals, *output_distortions, *output_energy;
		int deformed_size = deformed.size();
		pointer_of_lhs[1] = mxCreateDoubleMatrix(deformed_size * 2, 1, mxREAL);
		output_grad = mxGetPr(pointer_of_lhs[1]);

		pointer_of_lhs[2] = mxCreateDoubleMatrix(deformed_size * 2, 1, mxREAL);
		output_search_dir = mxGetPr(pointer_of_lhs[2]);

		pointer_of_lhs[3] = mxCreateDoubleMatrix(tri_num, simplex_dim, mxREAL);
		output_sing_vals = mxGetPr(pointer_of_lhs[3]);

		pointer_of_lhs[4] = mxCreateDoubleMatrix(tri_num, 1, mxREAL);
		output_distortions = mxGetPr(pointer_of_lhs[4]);

		pointer_of_lhs[5] = mxCreateDoubleMatrix(1, 1, mxREAL);
		output_energy = mxGetPr(pointer_of_lhs[5]);
		output_energy[0] = 0;

		if (solverSpec.report_data){ 
			model.return_search_dir = true;
			(*updated) = *(original);
			solverSpec.solver_num = 0;
			//std::cout << " \n Recompute distortions for statistics:";
			OptimizeABCD(solverSpec, model,
				gradient, search_direction,
				original, updated, updated_copy,
				all_elements, free_vertices, fixed_vertices,
				single_block_color
			);
			model.return_search_dir = false;
		}

		for (size_t i = 0; i < tri_num; i++) {
			output_sing_vals[0 + i] = model.svd_s_[i][0];
			output_sing_vals[tri_num + i] = model.svd_s_[i][1];
			output_distortions[i] = model.element_distortion[i];
			output_energy[0] += model.volume_[i] * model.element_distortion[i];
		}

		for (size_t i = 0; i < deformed_size; i++) {
			output_search_dir[simplex_dim * i + 0] = search_direction[2 * i];
			output_search_dir[simplex_dim * i + 1] = search_direction[2 * i + 1];
			output_grad[simplex_dim * i + 0] = gradient[2 * i];
			output_grad[simplex_dim * i + 1] = gradient[2 * i + 1];
		}
	}
	runtime_seconds[RunTime::OutputSave] = double(clock() - begin_output_save) / CLOCKS_PER_SEC;
	//std::cout << "\n -Output save time=" << runtime_seconds[RunTime::OutputSave];

	double total_runtime_seconds = 0;
	for (int i = RunTime::InputUpload; i <= RunTime::OutputSave;i++)
		total_runtime_seconds += runtime_seconds[i];
	

	if (num_of_lhs >= 7) {
		int run_times_num = RunTime::OutputSave - RunTime::InputUpload+1;
		pointer_of_lhs[6] = mxCreateDoubleMatrix(run_times_num, 1, mxREAL);
		double *run_time_meausres_ptr = mxGetPr(pointer_of_lhs[6]);
		for (int i = RunTime::InputUpload; i <= RunTime::OutputSave;i++){
			run_time_meausres_ptr[i] = runtime_seconds[i];
		}
	}

	return;
}
