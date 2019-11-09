// Copyright @2019. All rights reserved.
// Authors: anaitsat@campus.technion.ac.il (Alex Naitsat)

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

	class DebugMex {
	public:
		bool ConditionByName(const std::string& var_name = "") {
			bool is_output = true;
			return is_output;
		}

		void Mat2Mfile(const Eigen::MatrixXd& M, const std::string& M_name);
		void MatInt2Mfile(const Eigen::MatrixXi& M, const std::string& M_name);
		void Mat2Mfile(const Eigen::VectorXd& V, const std::string& V_name = "",
			const std::vector<int>& index = {});

		void ColumnVec2Mat(const Eigen::VectorXd& V, const std::string& V_name = "",
			const std::vector<int>& index = {}, int d = 2);

		void MatInt2Mfile(const Eigen::VectorXi& V, const std::string& V_name = "");

		void Mat2Mfile(const std::vector<Eigen::VectorXd>& V, const std::string& V_name);
		void Mat2Mfile(const std::vector<Eigen::Vector2d>& V,
			const std::string& V_name,
			const std::vector<int>& index = {});
		
		void Mat2Mfile(const std::vector<Eigen::Vector2i>& V,
			const std::string& V_name,
			const std::vector<int>& index = {});

		void MatList2Mfile(const std::vector<Eigen::VectorXi>& V, const std::string& V_name);
		void MatList2Mfile(const std::vector<std::vector<int>>& V, const std::string& V_name);
		void MatList2Mfile(const std::vector<std::set<int>>& V, const std::string& V_name);
		std::vector<int> ColumnStackIndex(const std::vector<int>& index, int d = 2);

		void MatList2Mfile(const std::vector<Eigen::Matrix2d>& V, const std::string& V_name);
		void MatList2Mfile(const std::vector<Eigen::Matrix2d>& V, const std::string& V_name, const std::vector<int>& index);

		template <typename  T>
		void Vec2Mfile(const std::vector<T>& V, const std::string& M_name) {
			m_file << "\n" << M_name << " = [";
			m_file << V;
			m_file << " ];" << std::endl;
			PostOutput();
		}

		void SparseMat2Mfile(const std::vector<Eigen::Triplet<double>>& M, int dim,  const std::string& M_name);

		void SetArraty2Mfile(const std::set<int>* V, int array_len, const std::string& M_name);
		void SetArraty2Mfile(const std::vector<std::set<int>>& V, const std::string& M_name);

		bool debug_step_mode;
		std::ofstream m_file;
		std::string m_file_name;

		void SetMatfile(std::string& file_name) {
			if (m_file.is_open()) Close();

			m_file.open(file_name);
			m_file_name = file_name;
		}

		DebugMex(const std::string file_name = "") {
			if (!file_name.empty())
				m_file.open(file_name);

			m_file_name = file_name;
			debug_step_mode = false;
		};

		~DebugMex() {
			Close();
		};

		void Close() {
			if (m_file.is_open())
				m_file.close();

			std::cout << "\nClosing " << m_file_name;
		}

	private:
		void PostOutput() {
			if (debug_step_mode && m_file.is_open())
				m_file.flush();
		}
	};

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
		int max_block_iterations = 1;				   
		int max_global_iterations = 100;               
		bool is_signed_svd = true;                     
		bool is_parallel = false;
		bool  single_fixed_block = false; 
		bool  is_global = false;
													   
		bool is_flip_barrier = false;
		double line_search_interval = 0.5;             
		double constant_step_size  = 0;                
		double zero_grad_eps = 1e-18;				   
		double K_hat = 1.0;                                    
		bool   report_data = true; 
		bool is_distortion_data_updated = false; 
	};
}
