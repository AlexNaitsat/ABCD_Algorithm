// Copyright @2019. All rights reserved.
// Authors: anaitsat@campus.technion.ac.il (Alexander Naitsat)

#pragma once

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mex.h>
#include <set>
#include <queue> 


#pragma once

namespace data_io {
	mxArray* VectorOfVectorsToCellArray(const std::vector < std::vector<int>>& vector_of_vecors);
	mxArray* VectorOfSetsToCellArray   (const std::vector < std::set<int>>& vector_of_vecors);

	bool FieldDataToPointer(const mxArray* struct_pnt, 
							const char* field_name, double** out_data, 
							int* M =NULL, int *N=NULL,			
							int index = 0);

	template <typename T>
	bool FieldDataToArray(const mxArray* struct_pnt, const char* field_name,
		T* out_data, size_t data_size, int index = 0) {
		const mxArray* field_array_ptr = mxGetField(struct_pnt, index, field_name);
		if (field_array_ptr == NULL) {
			return false;
			std::cout << "\nNo " << field_name << "field" << std::endl;
		}

		double* data_ptr = mxGetPr(field_array_ptr);
		for (size_t i = 0; i < data_size; i++)
			out_data[i] = data_ptr[i];
		return true;
	}

	template <typename T>
	bool FieldDataToArray(const mxArray* struct_pnt, const char* field_name,
		std::vector<T>& out_data, size_t data_size, int index = 0) {
		const mxArray* field_array_ptr = mxGetField(struct_pnt, index, field_name);
		if (field_array_ptr == NULL) {
			std::cout << "\nNo " << field_name << " field" << std::endl;
			return false;
		}

		double* data_ptr = mxGetPr(field_array_ptr);
		for (size_t i = 0; i < data_size; i++)
			out_data[i] = data_ptr[i];

		return true;
	}

	template <typename T>
	bool FieldCellArrayToVectorArray(const mxArray* struct_pnt, const char* field_name,
		std::vector<std::vector<T>>& out_data, int index = 0) {

		const mxArray* field_array_ptr = mxGetField(struct_pnt, index, field_name);
		if (field_array_ptr == NULL) {
			std::cout << "\nNo " << field_name << " field" << std::endl;
			return false;
		}
		if  (!mxIsCell(field_array_ptr)) {
			std::cout << "\n Field " << field_name << " is not cell array" << std::endl;
			return false;
		}

		int block_num = std::max(mxGetM(field_array_ptr), mxGetN(field_array_ptr));

		out_data.resize(block_num);
		for (size_t bi = 0; bi < block_num; bi++) {
			mxArray *cellArrayElement = mxGetCell(field_array_ptr, bi);
			if (mxIsEmpty(cellArrayElement))
				continue; 
			int element_size = std::max(mxGetN(cellArrayElement), mxGetM(cellArrayElement));
			double  *elementBlock = mxGetPr(cellArrayElement);
			out_data[bi].resize(element_size);
			for (size_t j = 0; j < element_size; j++) {
				out_data[bi][j] = elementBlock[j];
			}
		}
		return true;
	}

	template <typename T>
	std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
	{
		for (auto it : v) {
			os << it<<", ";
		}
		return os;
	}


	template <typename T>
	std::ostream& operator<<(std::ostream& os, const std::set<T>& v)
	{
		for (auto it : v) {
			os << it << ", ";
		}
		return os;
	}


	struct SolverSpecification {
		int solver_num = 0;							   
		int energy_num = 0;							   
		size_t non_empty_block_num = 0;
		double max_block_iterations = 1;
		std::vector<double> block_iteration_range = std::vector<double>(2,0);//[min_iter,max_iter, step]
		int max_global_iterations = 1;               
		int cycle_num = 4;
		bool is_signed_svd = true;                     
		bool is_parallel =        false, is_parallel_grad = false,
			 is_parallel_energy = false, is_parallel_hessian = false;

		bool use_pardiso_solver = false;
		bool single_fixed_block = true; 
		bool is_global = false;
		bool verbose = false;
													   
		bool is_flip_barrier = false;
		double line_search_interval = 0.5;             
		double constant_step_size  = 0;                
		double zero_grad_eps = 1e-18;				   
		double  K_hat = 1.0; 
		std::queue<double> K_hat_queue;
		int K_hat_size = 0;

		bool   report_data = true; 
		bool is_distortion_data_updated = false; 
		double min_distortion =0;
		double energy_tolerance = 1e-3;
		std::vector <double> invalid_penalty;
		void update_block_iteration() {
				max_block_iterations = std::min(max_block_iterations + block_iteration_range[2],
					block_iteration_range[1]);
		}
	};
}
