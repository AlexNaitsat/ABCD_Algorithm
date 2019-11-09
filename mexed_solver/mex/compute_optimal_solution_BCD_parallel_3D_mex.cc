// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#include "stdafx.h"

#include <iostream>
#include <vector>
#include <mex.h>
#include <Eigen/Dense>
#include "common/optimization/line_search_util.h"
#include "common/optimization/stop_check_util.h"
#include "common/solver/eigen/eigen_solver.h"
#include "energy_model/distortion_kernel/arap_kernel_3d.h"
#include "energy_model/distortion_kernel/symmetric_dirichlet_kernel_3d.h"
#include "energy_model/mesh_distortion/isotropic_svd_energy_model_3d.h"
#include <fstream>
#include <string>
#include "mat.h"

#include <omp.h>
#include <chrono>

#include "data_io/data_io_utils.h"
using namespace data_io;

#define EXE_MEX_TEST_PROJECT 1
void GD_SearchDirectionInBlock(std::vector<Eigen::Vector3d> &original,
							   mesh_distortion::IsotropicSVDEnergyModel3D &model,
							   std::vector<Eigen::Vector3d> *updated,
							   SolverSpecification &solverSpec,
							   Eigen::VectorXd &gradient,
							   Eigen::VectorXd &search_direction,
							   const std::vector<int> &element_block,
							   const std::vector<int> &free_vertex_block,
							   const std::vector<int> &bnd_vertex_block)
{

	assert(updated != nullptr);

	for (auto vi : free_vertex_block)
	{
		search_direction[3 * vi] = -gradient[3 * vi];
		search_direction[3 * vi + 1] = -gradient[3 * vi + 1];
		search_direction[3 * vi + 2] = -gradient[3 * vi + 2];
	}
}

void PN_SearchDirectionInBlock(std::vector<Eigen::Vector3d> &original,
							   mesh_distortion::IsotropicSVDEnergyModel3D &model,
							   std::vector<Eigen::Vector3d> *updated,
							   SolverSpecification &solverSpec,
							   Eigen::VectorXd &gradient,
							   Eigen::VectorXd &search_direction,
							   const std::vector<int> &element_block,
							   const std::vector<int> &free_vertex_block,
							   const std::vector<int> &bnd_vertex_block)
{
	assert(updated != nullptr);
	std::vector<common::solver::eigen::EigenSolver::EigenEntry> entry_list;
	int ver_num = free_vertex_block.size();
	common::solver::eigen::EigenSolver m_solver;

	model.ComputeHessianNonzeroEntriesDirConstraintsInBlock(original, &entry_list,
															element_block);
	m_solver.SetPattern(entry_list, 3 * ver_num);
	m_solver.AnalyzePattern();
	m_solver.Factorize();
	m_solver.SolveDirichletConstraints(-gradient, &search_direction, free_vertex_block, bnd_vertex_block);
}

void UpdateSolutionInBlock(std::vector<Eigen::Vector3d> &original,
						   mesh_distortion::IsotropicSVDEnergyModel3D &model,
						   std::vector<Eigen::Vector3d> *updated,
						   SolverSpecification &solverSpec,
						   Eigen::VectorXd &gradient,
						   Eigen::VectorXd &search_direction,
						   const std::vector<int> &element_block,
						   const std::vector<int> &free_vertex_block,
						   const std::vector<int> &bnd_vertex_block)
{

	if (!free_vertex_block.size())
	{
		std::cout << "\n block without free vertices";
		return;
	}

	int ver_num = free_vertex_block.size();
	for (auto vi : bnd_vertex_block)
	{
		search_direction(3 * vi) = 0;
		search_direction(3 * vi + 1) = 0;
		search_direction(3 * vi + 2) = 0;
		(*updated)[vi] = original[vi];
		model.blockFreeVertIndex[vi] = -1;
	}

	for (size_t i = 0; i < ver_num; i++)
	{
		int v = free_vertex_block[i];
		model.blockFreeVertIndex[v] = i;
	}

	for (int block_iter = 0; block_iter < solverSpec.max_block_iterations; block_iter++)
	{
		double fk = model.ComputeEnergyInBlock(original, element_block);
		model.ComputeGradientInBlock(original, &gradient, element_block, free_vertex_block);
		double grad_sq_norm = 0;
		for (auto vi : free_vertex_block)
		{
			double vi_gradient[] = {gradient[3 * vi], gradient[3 * vi + 1], gradient[3 * vi + 2]};
			grad_sq_norm += vi_gradient[0] * vi_gradient[0] + vi_gradient[1] * vi_gradient[1] + vi_gradient[2] * vi_gradient[2];
		}

		if (grad_sq_norm < 1e-16)
		{
			if (!block_iter)
			{
				if (!model.return_search_dir)
					for (auto vi : free_vertex_block)
						(*updated)[vi] = original[vi];
				else
					for (auto vi : free_vertex_block)
					{
						search_direction(3 * vi) = 0;
						search_direction(3 * vi + 1) = 0;
						search_direction(3 * vi + 2) = 0;
					}
			}
			break;
		}

		switch (solverSpec.solver_num)
		{
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
		if (model.return_search_dir)
			return;

		double step_time;
		if (solverSpec.constant_step_size > 0)
			step_time = solverSpec.constant_step_size;
		
		double reduced_energy =
			common::optimization::ArmijoLineSearchInBlock3D(
				original, step_time * search_direction, model, updated,
				gradient, fk, element_block, free_vertex_block);

		if (reduced_energy > fk)
		{
			for (auto vi : free_vertex_block)
				(*updated)[vi] = original[vi];
			return;
		}

		for (auto vi : free_vertex_block)
			original[vi] = (*updated)[vi];
	}
}

void mexFunction(int num_of_lhs,
				 mxArray *pointer_of_lhs[],
				 int num_of_rhs,
				 const mxArray *pointer_of_rhs[])
{

	double *position_rest, *position_deformed, *mesh, *num_of_vertex,
		*num_of_element, *kernel_type, *spd_threshold, *max_iteration,
		*solver_number, *energy_spec;

	const mxArray *element_blocks_pntr, *free_vertex_blocks_pntr,
		*bnd_vertex_blocks_pntr, *blocks_by_color_ptr = NULL;

	position_rest = mxGetPr(pointer_of_rhs[0]);
	position_deformed = mxGetPr(pointer_of_rhs[1]);
	mesh = mxGetPr(pointer_of_rhs[2]);
	num_of_vertex = mxGetPr(pointer_of_rhs[3]);
	num_of_element = mxGetPr(pointer_of_rhs[4]);

	SolverSpecification solverSpec;
	const mxArray *struct_pnt = pointer_of_rhs[5];
	if (mxGetClassID(struct_pnt) != mxSTRUCT_CLASS)
		mexErrMsgTxt("9th parameter should be struct");

	std::vector<double> invalid_penalty(3, 0);
	double spd_thresh = 1e-6;

	FieldDataToArray<int>(struct_pnt, "solver_num", &solverSpec.solver_num, 1);
	FieldDataToArray<int>(struct_pnt, "energy_num", &solverSpec.energy_num, 1);
	FieldDataToArray<double>(struct_pnt, "invalid_penalty", invalid_penalty, 3);
	FieldDataToArray<int>(struct_pnt, "max_block_iterations", &solverSpec.max_block_iterations, 1);
	FieldDataToArray<int>(struct_pnt, "max_global_iterations", &solverSpec.max_global_iterations, 1);
	FieldDataToArray<bool>(struct_pnt, "is_signed_svd", &solverSpec.is_signed_svd, 1);
	FieldDataToArray<bool>(struct_pnt, "is_parallel", &solverSpec.is_parallel, 1);
	FieldDataToArray<bool>(struct_pnt, "is_flip_barrier", &solverSpec.is_flip_barrier, 1);

	FieldDataToArray<double>(struct_pnt, "constant_step_size", &solverSpec.constant_step_size, 1);
	int source_dim = 2;
	FieldDataToArray<int>(struct_pnt, "source_dim", &source_dim, 1);

	if (!(mxIsCell(pointer_of_rhs[6]) && mxIsCell(pointer_of_rhs[7]) &&
		  mxIsCell(pointer_of_rhs[8])))
	{
		mexErrMsgTxt("7h-9th input variable must be cell arrays");
	}
	element_blocks_pntr = pointer_of_rhs[6];
	free_vertex_blocks_pntr = pointer_of_rhs[7];
	bnd_vertex_blocks_pntr = pointer_of_rhs[8];

	int tet_num = num_of_element[0];
	int ver_num = num_of_vertex[0];
	int kernel_t = solverSpec.energy_num;
	int block_num = std::max(mxGetM(element_blocks_pntr), mxGetN(element_blocks_pntr));
	int color_num = 0;
	int max_iter_num = solverSpec.max_global_iterations;

	Eigen::MatrixXd rest(ver_num, source_dim);

	std::vector<Eigen::Vector4i> tetrahedron(tet_num);
	std::vector<Eigen::Vector3d> deformed(ver_num);
	std::vector<std::vector<int>> element_blocks(block_num), free_vertex_blocks(block_num),
		bnd_vertex_blocks(block_num), blocks_by_color(color_num);

	for (size_t i = 0; i < tet_num; i++)
	{
		tetrahedron[i][0] = mesh[i + 0 * tet_num] - 1;
		tetrahedron[i][1] = mesh[i + 1 * tet_num] - 1;
		tetrahedron[i][2] = mesh[i + 2 * tet_num] - 1;
		tetrahedron[i][3] = mesh[i + 3 * tet_num] - 1;
	}

	for (size_t i = 0; i < ver_num; i++)
	{
		rest(i, 0) = position_rest[i + 0 * ver_num];
		rest(i, 1) = position_rest[i + 1 * ver_num];
		rest(i, 2) = position_rest[i + 2 * ver_num];

		deformed[i][0] = position_deformed[i + 0 * ver_num];
		deformed[i][1] = position_deformed[i + 1 * ver_num];
		deformed[i][2] = position_deformed[i + 2 * ver_num];
	}

	for (size_t bi = 0; bi < block_num; bi++)
	{
		mxArray *elementBlockArray = mxGetCell(element_blocks_pntr, bi),
				*freeBlockArray = mxGetCell(free_vertex_blocks_pntr, bi),
				*bndBlockArray = mxGetCell(bnd_vertex_blocks_pntr, bi);

		double *elementBlock = mxGetPr(elementBlockArray),
			   *freeBlock = mxGetPr(freeBlockArray),
			   *bndBlock = mxGetPr(bndBlockArray);

		int element_block_size = mxGetN(elementBlockArray),
			free_block_size = mxGetN(freeBlockArray),
			bnd_block_size = mxGetN(bndBlockArray);
		element_blocks[bi].assign(element_block_size, 0);

		free_vertex_blocks[bi].assign(free_block_size, 0);

		bnd_vertex_blocks[bi].assign(bnd_block_size, 0);

		for (size_t j = 0; j < element_block_size; j++)
		{
			element_blocks[bi][j] = elementBlock[j];
		}
		for (size_t j = 0; j < free_block_size; j++)
			free_vertex_blocks[bi][j] = freeBlock[j];
		for (size_t j = 0; j < bnd_block_size; j++)
			bnd_vertex_blocks[bi][j] = bndBlock[j];
	}

	distortion_kernel::DistortionKernel3D *kernel = NULL;

	switch (solverSpec.energy_num)
	{
	case 0:
		std::cout << "\n ARAP energy";
		kernel = new distortion_kernel::ARAPKernel3D();
		break;
	case 1:
		std::cout << "\n Symmetric Dirichlet energy";
		kernel = new distortion_kernel::SymmetricDirichletKernel3D();
		break;
	default:
		std::cout << "\n Input energy number " << solverSpec.energy_num << " is not supported ";
		mexErrMsgTxt("\n Unsupported energy");
		return;
		break;
	}

	mesh_distortion::IsotropicSVDEnergyModel3D model;
	model.SetRestMesh(rest, tetrahedron);
	model.SetDistortionKernel(kernel);
	model.SetEnforcingSPD(true, spd_thresh);
	model.SetSignedSVD(solverSpec.is_signed_svd);

	FieldDataToArray<double>(struct_pnt, "ls_interval", &model.ls_interval, 1);
	FieldDataToArray<double>(struct_pnt, "ls_alpha", &model.ls_alpha, 1);
	FieldDataToArray<double>(struct_pnt, "ls_beta", &model.ls_beta, 1);
	FieldDataToArray<bool>(struct_pnt, "is_flip_barrier", &solverSpec.is_flip_barrier, 1);
	FieldDataToArray<bool>(struct_pnt, "return_search_dir", &model.return_search_dir, 1);

	std::vector<Eigen::Vector3d> buffer = deformed, updated_copy = deformed;

	std::vector<Eigen::Vector3d> *original = &deformed;
	std::vector<Eigen::Vector3d> *updated = &buffer;

	std::cout << "\nnumber of blocks =" << block_num << ","
			  << solverSpec.max_block_iterations << " block iterations";

	Eigen::VectorXd search_direction(3 * ver_num),
		gradient(3 * ver_num);
	search_direction.setZero();
	gradient.setZero();

	if (num_of_rhs >= 10)
	{
		blocks_by_color_ptr = pointer_of_rhs[9];
		color_num = std::max(mxGetM(blocks_by_color_ptr), mxGetN(blocks_by_color_ptr));
		blocks_by_color.resize(color_num);
		for (size_t ci = 0; ci < color_num; ci++)
		{
			mxArray *colorBlockArr = mxGetCell(blocks_by_color_ptr, ci);
			double *colorBlock = mxGetPr(colorBlockArr);
			int color_block_size = mxGetN(colorBlockArr);
			blocks_by_color[ci].resize(color_block_size, 0);
			for (size_t j = 0; j < color_block_size; j++)
			{
				blocks_by_color[ci][j] = colorBlock[j];
			}
		}
	}

	bool is_global_solver = (block_num <= 1);
	std::map<int, bool> thread_calls;

	model.SetDirichletConstraints(ver_num, -1);

	if (solverSpec.is_parallel && color_num)
	{
		std::cout << "\n Running in parallel\n";
	}
	else
	{
		std::cout << "\n Running sequentially";
		if (color_num)
			std::cout << " with block coloring order";
		std::cout << std::endl;
	}
	for (int i = 0; i < max_iter_num; i++)
	{
		std::cout << ((solverSpec.solver_num) ? "PN" : "GD") << " Iteration " << i << std::endl;
		if (color_num)
		{
			for (int ci = 0; ci < color_num; ci++)
			{
				int parallel_block_num = blocks_by_color[ci].size();
#pragma omp parallel for schedule(dynamic) if (solverSpec.is_parallel)
				for (int i = 0; i < parallel_block_num; i++)
				{
					int bi = blocks_by_color[ci][i];
					thread_calls[omp_get_thread_num()] = 1;

					UpdateSolutionInBlock(updated_copy, model, updated, solverSpec,
										  gradient, search_direction, //dfk,
										  element_blocks[bi], free_vertex_blocks[bi], bnd_vertex_blocks[bi]);
				}
			}
		}
		else
		{
			for (int bi = 0; bi < block_num; bi++)
			{
				UpdateSolutionInBlock(updated_copy, model, updated, solverSpec,
									  gradient, search_direction,
									  element_blocks[bi], free_vertex_blocks[bi], bnd_vertex_blocks[bi]);
			}
		}

		if (common::optimization::IsNumericallyConverged3D(*original, *updated))
		{
			//std::cout << "\n stop due to the convergence criteria";
			break;
		}

		if (!model.return_search_dir)
		{
			std::vector<Eigen::Vector3d> *temp = original;
			original = updated;
			updated = temp;
		}
	}

	std::cout << "\nThreads used :";
	for (auto ti : thread_calls)
		std::cout << ti.first << " ,";

	if (!model.return_search_dir)
	{
		double *output;
		pointer_of_lhs[0] = mxCreateDoubleMatrix(deformed.size() * 3, 1, mxREAL);
		output = mxGetPr(pointer_of_lhs[0]);

		for (size_t i = 0; i < deformed.size(); i++)
		{
			output[3 * i + 0] = (*original)[i][0];
			output[3 * i + 1] = (*original)[i][1];
			output[3 * i + 2] = (*original)[i][2];
		}
	}
	else
	{
		pointer_of_lhs[0] = mxCreateDoubleMatrix(1, 0, mxREAL);
	}
	if (num_of_lhs == 6)
	{
		const int simplex_dim = 3;
		std::cout << "\n Returns search direction";
		double *output_grad, *output_search_dir, *output_sing_vals, *output_distortions, *output_energy;
		int deformed_size = deformed.size();
		pointer_of_lhs[1] = mxCreateDoubleMatrix(deformed_size * 3, 1, mxREAL);
		output_grad = mxGetPr(pointer_of_lhs[1]);

		pointer_of_lhs[2] = mxCreateDoubleMatrix(deformed_size * 3, 1, mxREAL);
		output_search_dir = mxGetPr(pointer_of_lhs[2]);

		pointer_of_lhs[3] = mxCreateDoubleMatrix(tet_num, simplex_dim, mxREAL);
		output_sing_vals = mxGetPr(pointer_of_lhs[3]);

		pointer_of_lhs[4] = mxCreateDoubleMatrix(tet_num, 1, mxREAL);
		output_distortions = mxGetPr(pointer_of_lhs[4]);

		pointer_of_lhs[5] = mxCreateDoubleMatrix(1, 1, mxREAL);
		output_energy = mxGetPr(pointer_of_lhs[5]);
		output_energy[0] = 0;

		for (size_t i = 0; i < tet_num; i++)
		{
			output_sing_vals[0 + i] = model.svd_s_[i][0];
			output_sing_vals[tet_num + i] = model.svd_s_[i][1];
			output_distortions[i] = model.element_distortion[i];

			output_energy[0] += model.volume_[i] * model.element_distortion[i];
		}

		for (size_t i = 0; i < deformed_size; i++)
		{
			output_search_dir[simplex_dim * i + 0] = search_direction[3 * i];
			output_search_dir[simplex_dim * i + 1] = search_direction[3 * i + 1];
			output_search_dir[simplex_dim * i + 2] = search_direction[3 * i + 2];

			output_grad[simplex_dim * i + 0] = gradient[3 * i];
			output_grad[simplex_dim * i + 1] = gradient[3 * i + 1];
			output_grad[simplex_dim * i + 2] = gradient[3 * i + 2];
		}
	}
	//std::cout << "\n finished";
	return;
}
