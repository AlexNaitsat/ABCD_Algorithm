// Copyright @2020. All rights reserved.
// Authors: anaitsat@campus.technion.ac.il  (Alexander Naitsat),
//          mike323zyf@gmail.com (Yufeng Zhu)

#include <iostream>
#include <set>
#include <vector>

#include <Eigen/Core>
#include "stdafx.h"

#include <iostream>
#include <vector>
#include <mex.h>
#include <Eigen/Dense>
#include "energy_model/distortion_kernel/arap_kernel_2d.h"
#include "energy_model/distortion_kernel/flip_penalty_kernel_2d.h"					 
//#include "energy_model/distortion_kernel/symmetric_dirichlet_kernel_2d.h"
#include "energy_model/distortion_kernel/sym_dirichlet_filtered_kernel_2d.h"
//#include "energy_model/distortion_kernel/symmetric_dirichlet_kernel_3d.h"

#include "energy_model/distortion_kernel/arap_kernel_3d.h"
#include "energy_model/distortion_kernel/flip_penalty_kernel_3d.h"	
#include "energy_model/distortion_kernel/sym_dirichlet_filtered_kernel_3d.h"

#include "energy_model/mesh_distortion/isotropic_svd_energy_model_2d.h"
#include <fstream>
#include <string>
#include "mat.h"
#include <omp.h>
#include <time.h>
#include <chrono>
#include "data_io/data_io_utils.h"
#include "ABCD_optimization_core_2d.h"

#include <algorithm>
#include <iterator>

using namespace data_io;
enum RunTime { InputUpload, SearchDirComputation, BlockProcessing, RestInitialization, Optimization, OutputSave };

void GetMeshConnectivityData2D(const std::vector<Eigen::Vector3i> &triangle,
	size_t vn,
	std::vector<std::set<int>>& v4v,
	std::vector<std::set<int>>& t4v,
	std::vector<Eigen::Vector2i>& eu,
	std::vector<bool>&  is_bnd_v,
	bool is_1_based_index=false)//if is_1_based_index=1 then index start from 1 instead of 0
{
	int  tn = triangle.size();
	eu.clear();
	eu.reserve(3 * vn);	

	v4v.resize(vn);
	t4v.resize(vn);


	is_bnd_v.resize(vn, false);

	for (int i = 0; i < tn; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			int va = triangle[i][j]           + is_1_based_index;
			int vb = triangle[i][(j + 1) % 3] + is_1_based_index;
			if (va < vb)
				eu.push_back(Eigen::Vector2i(va, vb));
			else 
				eu.push_back(Eigen::Vector2i(vb, va));
		}
	}

	auto edge_compare = [](const Eigen::Vector2i &ea, const Eigen::Vector2i &eb) {
		if (ea[0] == eb[0])
		{
			return ea[1] < eb[1];
		}
		return ea[0] < eb[0];
	};

	sort(eu.begin(), eu.end(), edge_compare);

	for (int i = 1; i < eu.size(); ++i)
	{
		if (std::abs(eu[i - 1][0]) == eu[i][0] &&
			std::abs(eu[i - 1][1]) == eu[i][1])
		{
			eu[i][0] *= -1;
			eu[i][1] *= -1;
		}
	}

	eu.erase(std::remove_if(eu.begin(), eu.end(),
		[](const Eigen::Vector2i &edge) {
		return edge[0] < 0 || edge[1] < 0;
	}),
		eu.end());

	for (int i = 0; i < triangle.size(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			v4v[triangle[i][j]].insert(triangle[i][(j + 1) % 3] + is_1_based_index);
			v4v[triangle[i][j]].insert(triangle[i][(j + 2) % 3] + is_1_based_index);
			t4v[triangle[i][j]].insert(i + is_1_based_index);
		}
	}

	auto set_intersect = [](const std::set<int> &set_a, const std::set<int> &set_b) {
		std::set<int> intersection;
		std::set_intersection(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
			std::inserter(intersection, intersection.begin()));
		return intersection.size();
	};

	for (int i = 0; i < eu.size(); ++i)
	{
		const auto &neighbor_a = t4v[eu[i][0] -is_1_based_index];
		const auto &neighbor_b = t4v[eu[i][1] -is_1_based_index];

		if (set_intersect(neighbor_a, neighbor_b) == 1)
		{
			is_bnd_v[eu[i][0] - is_1_based_index] = true;
			is_bnd_v[eu[i][1] - is_1_based_index] = true;
		}
	}
}



	void GetMeshConnectivityData3D(const std::vector<Eigen::Vector4i> &tet,
									size_t vn,
									std::vector<std::set<int>>& v4v,
									std::vector<std::set<int>>& t4v,
									std::vector<Eigen::Vector2i>& eu,
									std::vector<bool>&  is_bnd_v,
									bool is_1_based_index = false)
	{
		int  tn = tet.size();
		eu.clear();
		eu.reserve(6 * vn);
		v4v.resize(vn);
		t4v.resize(vn);

		is_bnd_v.resize(vn, false);

		for (int i = 0; i < tn; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				int va = tet[i][j] + is_1_based_index;
				int vb = tet[i][(j + 1) % 4] + is_1_based_index;
				if (va < vb)
					eu.push_back(Eigen::Vector2i(va, vb));
				else
					eu.push_back(Eigen::Vector2i(vb, va));
			}
		}

		auto edge_compare = [](const Eigen::Vector2i &ea, const Eigen::Vector2i &eb) {
			if (ea[0] == eb[0])
			{
				return ea[1] < eb[1];
			}
			return ea[0] < eb[0];
		};

		sort(eu.begin(), eu.end(), edge_compare);

		for (int i = 1; i < eu.size(); ++i)
		{
			if (std::abs(eu[i - 1][0]) == eu[i][0] &&
				std::abs(eu[i - 1][1]) == eu[i][1])
			{
				eu[i][0] *= -1;
				eu[i][1] *= -1;
			}
		}

		eu.erase(std::remove_if(eu.begin(), eu.end(),
			[](const Eigen::Vector2i &edge) {
			return edge[0] < 0 || edge[1] < 0;
		}),
			eu.end());

		for (int i = 0; i < tet.size(); ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				v4v[tet[i][j]].insert(tet[i][(j + 1) % 4] + is_1_based_index);
				v4v[tet[i][j]].insert(tet[i][(j + 2) % 4] + is_1_based_index);
				v4v[tet[i][j]].insert(tet[i][(j + 3) % 4] + is_1_based_index);
				t4v[tet[i][j]].insert(i + is_1_based_index);
			}
		}

		auto set_intersect = [](const std::set<int> &set_a, const std::set<int> &set_b) {
			std::set<int> intersection;
			std::set_intersection(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
				std::inserter(intersection, intersection.begin()));
			return intersection.size();
		};

		for (int i = 0; i < eu.size(); ++i)
		{
			const auto &neighbor_a = t4v[eu[i][0] - is_1_based_index];
			const auto &neighbor_b = t4v[eu[i][1] - is_1_based_index];

			if (set_intersect(neighbor_a, neighbor_b) == 1)
			{
				is_bnd_v[eu[i][0] - is_1_based_index] = true;
				is_bnd_v[eu[i][1] - is_1_based_index] = true;
			}
		}
	}

// matlab interface: GetMeshData_mex(T, V, tn,vn,is_1_based_index, <fV>, <solver_spec>)
void mexFunction(int num_of_lhs,
	mxArray* pointer_of_lhs[],
	int num_of_rhs,
	const mxArray* pointer_of_rhs[]) 
{
	clock_t data_initialization_begin = clock();

	//Inputs pointer init.
	const mxArray   *ptr_mesh              = pointer_of_rhs[0], //mandotory inputs
					*ptr_position_rest     = pointer_of_rhs[1],
					*ptr_num_of_element    = pointer_of_rhs[2],
					*ptr_num_of_vertex     = pointer_of_rhs[3],
					*ptr_is_1_based_index  = pointer_of_rhs[4],
					*ptr_position_deformed = (num_of_rhs > 5) ? pointer_of_rhs[5] : NULL,//optional inputs 
					*struct_pnt			   = (num_of_rhs > 6) ? pointer_of_rhs[6] : NULL,
					*ptr_is_fixed_vertex   = (num_of_rhs > 7) ? pointer_of_rhs[7] : NULL;


	double *position_rest, *position_deformed, *mesh, *num_of_vertex,
		*num_of_element, *kernel_type, *max_iteration,
		*solver_number, *energy_spec;

	position_rest = mxGetPr(ptr_position_rest);

	mesh = mxGetPr(ptr_mesh);
	bool is_triangle_mesh = (mxGetN(ptr_mesh)==3);
	num_of_element = mxGetPr(ptr_num_of_element);
	num_of_vertex  = mxGetPr(ptr_num_of_vertex);
	//double*  is_1_based_index_ = false;
	bool is_1_based_index = (mxGetPr(ptr_is_1_based_index))[0];
	

	int tri_num = num_of_element[0];
	int ver_num = num_of_vertex[0];
	int source_dim = 3;

	int color_num = 0;

	std::vector<Eigen::Vector3i> triangle;
	std::vector<Eigen::Vector4i> tet;
	if (is_triangle_mesh) {
		triangle.resize(tri_num);
		for (size_t i = 0; i < tri_num; i++) {
			triangle[i][0] = mesh[i + 0 * tri_num] - 1;//converting matab 1-based indices to 0-based indices 
			triangle[i][1] = mesh[i + 1 * tri_num] - 1;
			triangle[i][2] = mesh[i + 2 * tri_num] - 1;
		}
	}
	else {
		std::cout << "\n Tetrahedral mesh"<<std::endl;
		tet.resize(tri_num);
		for (size_t i = 0; i < tri_num; i++) {
			tet[i][0] = mesh[i + 0 * tri_num] - 1;//converting matab 1-based indices to 0-based indices 
			tet[i][1] = mesh[i + 1 * tri_num] - 1;
			tet[i][2] = mesh[i + 2 * tri_num] - 1;
			tet[i][3] = mesh[i + 3 * tri_num] - 1;
		}
	}


	// compute mesh connenctivity data  (mandatory input/output) 
	double  *vv_mesh_edges;
	int mesh_edges_num = 0;
	std::vector<Eigen::Vector2i> unique_edges;
	std::vector<std::set<int>> v4v_set, t4v_set;
	std::vector<bool>  is_bnd_v;
	if (is_triangle_mesh) 
		GetMeshConnectivityData2D(triangle, ver_num, v4v_set, t4v_set,
							      unique_edges, is_bnd_v, is_1_based_index);
	 else 
		 GetMeshConnectivityData3D(tet, ver_num, v4v_set, t4v_set,
								   unique_edges, is_bnd_v, is_1_based_index);
	
	
	
	
	//mandatory output: 
	enum IndicexLHS {IsBnd_v=0, Eu, V4V, T4V, energy, elemnent_dist, sing_val, grad};

	pointer_of_lhs[IndicexLHS::IsBnd_v] = mxCreateDoubleMatrix(ver_num, 1, mxREAL);
	double *output_is_bnd_v = mxGetPr(pointer_of_lhs[IndicexLHS::IsBnd_v]);
	for (size_t v = 0; v < ver_num; v++) {
		output_is_bnd_v[v] = is_bnd_v[v];
	}

	pointer_of_lhs[IndicexLHS::Eu] = mxCreateDoubleMatrix(unique_edges.size() , 2, mxREAL);
	double *output_Eu = mxGetPr(pointer_of_lhs[IndicexLHS::Eu]);
	size_t edge_num = unique_edges.size();
	for (size_t i = 0; i <edge_num; i++) {
		output_Eu[0 + i]         = unique_edges[i][0];
		output_Eu[edge_num + i]  = unique_edges[i][1];
	}

	pointer_of_lhs[IndicexLHS::T4V] = VectorOfSetsToCellArray(t4v_set);
	pointer_of_lhs[IndicexLHS::V4V] = VectorOfSetsToCellArray(v4v_set);


	double data_initialization_end = clock();
	std::cout << "\nMesh data initialization: " <<
		 double(data_initialization_end - data_initialization_begin) / CLOCKS_PER_SEC <<" sec.";


	//=================== processing optional inputs
	//initialization of source and target coordinates
	if (!ptr_position_deformed) {
		return;
	}

	source_dim = mxGetN(ptr_position_rest);
	Eigen::MatrixXd rest(ver_num, source_dim);
	for (size_t i = 0; i < ver_num; i++) {
		rest(i, 0) = position_rest[i + 0 * ver_num];
		rest(i, 1) = position_rest[i + 1 * ver_num];
		if (source_dim == 3) {
			rest(i, 2) = position_rest[i + 2 * ver_num];
		}
	}


	std::vector<Eigen::Vector2d> deformed2D;
	std::vector<Eigen::Vector3d> deformed3D;
	
	position_deformed = mxGetPr(ptr_position_deformed);

	if (is_triangle_mesh) {
		deformed2D.resize(ver_num);
		for (size_t i = 0; i < ver_num; i++) {
			deformed2D[i][0] = position_deformed[i + 0 * ver_num];
			deformed2D[i][1] = position_deformed[i + 1 * ver_num];
		}
	}
	else {
		deformed3D.resize(ver_num);
		for (size_t i = 0; i < ver_num; i++) {
			deformed3D[i][0] = position_deformed[i + 0 * ver_num];
			deformed3D[i][1] = position_deformed[i + 1 * ver_num];
			deformed3D[i][2] = position_deformed[i + 2 * ver_num];
		}
	}

	std::vector<bool> is_fixed_vertex(ver_num,false);

	if (ptr_is_fixed_vertex) {
		double* is_fixed_vertex_double = mxGetPr(ptr_is_fixed_vertex);
		std::copy(is_fixed_vertex_double, is_fixed_vertex_double + ver_num, is_fixed_vertex.begin());
	}
	std::vector<std::vector<int>>	single_block_color(1, std::vector<int>{0});

	SolverSpecification solverSpec; //fixer and optimizer specs.


	double spd_thresh = 1e-6;
	if (mxGetClassID(struct_pnt) != mxSTRUCT_CLASS)
	{
		mexErrMsgTxt("9th and 10th  parameters should be struct");
	}
	solverSpec.invalid_penalty.resize(3, 0); 
	solverSpec.block_iteration_range.resize(3, 1);
	//solverSpec.spd_thresh = 1e-6;

	FieldDataToArray<int>(struct_pnt, "solver_num", &solverSpec.solver_num, 1);
	FieldDataToArray<int>(struct_pnt, "energy_num", &solverSpec.energy_num, 1);
	FieldDataToArray<int>(struct_pnt, "energy_num", &solverSpec.energy_num, 1);
	FieldDataToArray<int>(struct_pnt, "cycle_num", &solverSpec.cycle_num, 1);
	FieldDataToArray<bool>(struct_pnt, "is_global", &solverSpec.is_global, 1);
	FieldDataToArray<bool>(struct_pnt, "use_pardiso", &solverSpec.use_pardiso_solver, 1);

	FieldDataToArray<bool>(struct_pnt, "single_fixed_block", &solverSpec.single_fixed_block, 1);
	FieldDataToArray<double>(struct_pnt, "invalid_penalty", solverSpec.invalid_penalty, 3);
	FieldDataToArray<double>(struct_pnt, "block_iteration_range", solverSpec.block_iteration_range, 3);
	solverSpec.max_block_iterations = solverSpec.block_iteration_range[0];//initalizing block iteration 

	FieldDataToArray<int>(struct_pnt, "max_global_iterations", &solverSpec.max_global_iterations, 1);
	FieldDataToArray<bool>(struct_pnt, "is_signed_svd", &solverSpec.is_signed_svd, 1);
	FieldDataToArray<bool>(struct_pnt, "is_parallel", &solverSpec.is_parallel, 1);
	FieldDataToArray<bool>(struct_pnt, "is_parallel_grad", &solverSpec.is_parallel_grad, 1);
	FieldDataToArray<bool>(struct_pnt, "is_parallel_energy", &solverSpec.is_parallel_energy, 1);
	FieldDataToArray<bool>(struct_pnt, "is_parallel_hessian", &solverSpec.is_parallel_hessian, 1);

	FieldDataToArray<bool>(struct_pnt, "is_flip_barrier", &solverSpec.is_flip_barrier, 1);
	FieldDataToArray<bool>(struct_pnt, "report_data", &solverSpec.report_data, 1);

	FieldDataToArray<double>(struct_pnt, "constant_step_size", &solverSpec.constant_step_size, 1);

	FieldDataToArray<double>(struct_pnt, "zero_grad_eps", &solverSpec.zero_grad_eps, 1);
	FieldDataToArray<bool>(struct_pnt, "is_flip_barrier", &solverSpec.is_flip_barrier, 1);
	FieldDataToArray<double>(struct_pnt, "K_hat", &solverSpec.K_hat, 1);
	FieldDataToArray<double>(struct_pnt, "tolerance", &solverSpec.energy_tolerance, 1);


	distortion_kernel::DistortionKernel2D*   kernel2D    = {NULL};
	distortion_kernel::DistortionKernel3D*   kernel3D    = {NULL};

	bool is_inversion_free_stage_global = true; 

		switch (solverSpec.energy_num) {
		case 0:
			std::cout << "\measuring ARAP energy";
			if (is_triangle_mesh)
				kernel2D = new distortion_kernel::ARAPKernel2D();
			else
				kernel3D = new distortion_kernel::ARAPKernel3D();
			break;
		case 2:
			std::cout << "\nmeasuring Modified Symmetric Dirichlet energy";
			if (is_triangle_mesh) {
				kernel2D = new distortion_kernel::SymDirichletFilteredKernel2D();
				kernel2D->EnableFlipFilter(true);
			}
			else {
				kernel3D = new distortion_kernel::SymDirichletFilteredKernel3D();
				kernel3D->EnableFlipFilter(true);
			}

			break;
		case 3: 
			std::cout << "\nmeasuring Flip  Penalty energy, Lambda=" << solverSpec.invalid_penalty[0];
			if (is_triangle_mesh)
				kernel2D = new distortion_kernel::FlipPenaltyKernel2D(solverSpec.invalid_penalty);
			else 
				kernel3D = new distortion_kernel::FlipPenaltyKernel3D(solverSpec.invalid_penalty);
			break;
		default:
			std::cout << "\n Input energy number " << solverSpec.energy_num << " is not supported ";
			mexErrMsgTxt("\n Unsupported energy");
			return;
			break;

		}
	//----common model initialization ----
	mesh_distortion::IsotropicSVDEnergyModel2D model2D;
	mesh_distortion::IsotropicSVDEnergyModel3D model3D;

	if (is_triangle_mesh) {
		FieldDataToArray<bool>(struct_pnt, "is_flip_barrier", &model2D.is_flip_barrier, 1);
		model2D.SetRestMesh(rest, triangle);
		model2D.SetDirichletConstraints(ver_num, -1);
		model2D.SetDistortionKernel(kernel2D);
		model2D.SetEnforcingSPD(true, spd_thresh);
		model2D.SetSignedSVD(solverSpec.is_signed_svd);
	}
	else {
		FieldDataToArray<bool>(struct_pnt, "is_flip_barrier", &model3D.is_flip_barrier, 1);
		model3D.SetRestMesh(rest, tet);
		model3D.SetDirichletConstraints(ver_num, -1);
		model3D.SetDistortionKernel(kernel3D);
		model3D.SetEnforcingSPD(true, spd_thresh);
		model3D.SetSignedSVD(solverSpec.is_signed_svd);
	}

	// initializaing block relatated data 
	// for processing optional inputs/ outputs
	std::vector<std::vector<int>> all_elements(1), free_vertices(1), fixed_vertices(1);// dist. data optional output

	all_elements[0].resize(tri_num);
	for (auto iter= all_elements[0].begin(); iter != all_elements[0].end(); iter++)
	    *iter = (size_t) (iter - all_elements[0].begin());
	
	for (size_t v = 0; v < ver_num; v++) {
		if (is_fixed_vertex[v])
			fixed_vertices[0].push_back(v);
		else
			free_vertices[0].push_back(v);
	}
	
	//----------------------------------------------------------------------------------
		int simplex_dim = (is_triangle_mesh)? 2 : 3;
		double  *output_sing_vals, *output_distortions, *output_energy;
		int deformed_size = deformed2D.size();
		
		pointer_of_lhs[IndicexLHS::sing_val] = mxCreateDoubleMatrix(tri_num, simplex_dim, mxREAL);
		output_sing_vals					 = mxGetPr(pointer_of_lhs[IndicexLHS::sing_val]);

		pointer_of_lhs[IndicexLHS::elemnent_dist] = mxCreateDoubleMatrix(tri_num, 1, mxREAL);
		output_distortions						  = mxGetPr(pointer_of_lhs[IndicexLHS::elemnent_dist]);

		pointer_of_lhs[IndicexLHS::energy] = mxCreateDoubleMatrix(1, 1, mxREAL);
		output_energy					   = mxGetPr(pointer_of_lhs[IndicexLHS::energy]);
		output_energy[0] = 0;

		solverSpec.solver_num = 0;
		
		if (is_triangle_mesh) {
			model2D.return_search_dir = true;
			double fk = model2D.ComputeEnergyInBlock(deformed2D, all_elements[0], solverSpec, true);
		}
		else {
			model3D.return_search_dir = true;
			double fk = model3D.ComputeEnergyInBlock(deformed3D, all_elements[0], solverSpec, true);
		}

		if (is_triangle_mesh) {
			for (size_t i = 0; i < tri_num; i++) {
				output_sing_vals[0 + i]       = model2D.svd_s_[i][0];
				output_sing_vals[tri_num + i] = model2D.svd_s_[i][1];
				output_distortions[i]         = model2D.element_distortion[i];
				output_energy[0]             += model2D.volume_[i] * model2D.element_distortion[i];
			}
		} else {
			for (size_t i = 0; i < tri_num; i++) {
				output_sing_vals[  0       + i] = model3D.svd_s_[i][0];
				output_sing_vals[  tri_num + i] = model3D.svd_s_[i][1];
				output_sing_vals[2*tri_num + i] = model3D.svd_s_[i][2];
				output_distortions[i]           = model3D.element_distortion[i];
				output_energy[0]               += model3D.volume_[i] * model3D.element_distortion[i];
			}
		}

		//Additional dist. related parameters
		if (num_of_lhs >= IndicexLHS::grad + 1) {
			double *output_grad;
			pointer_of_lhs[IndicexLHS::grad] = mxCreateDoubleMatrix(deformed_size, simplex_dim, mxREAL);
			output_grad						 = mxGetPr(pointer_of_lhs[IndicexLHS::grad]);
			Eigen::VectorXd gradient(simplex_dim * ver_num); 
			gradient.setZero(); 

			if (is_triangle_mesh)
				model2D.ComputeGradientInBlock(deformed2D, &gradient, all_elements[0],
					free_vertices[0], solverSpec);
			else
				model3D.ComputeGradientInBlock(deformed3D, &gradient, all_elements[0],
					free_vertices[0], solverSpec);

			for (size_t i = 0; i < ver_num; i++)
				for (int j = 0; j < simplex_dim; j++)
					output_grad[j*ver_num + i] = gradient[simplex_dim * i + j];

		}
	
	std::cout <<", " <<double(clock() - data_initialization_end) / CLOCKS_PER_SEC<<" sec. \n";
	return;
}
