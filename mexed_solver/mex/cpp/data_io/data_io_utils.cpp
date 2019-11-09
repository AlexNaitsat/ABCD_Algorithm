// Copyright @2019. All rights reserved.
// Authors: anaitsat@campus.ac.il (Alex Naitsat)

#include "stdafx.h"
#include "data_io/data_io_utils.h"

namespace data_io {
	std::string iteration_prefix;

	mxArray* VectorOfVectorsToCellArray(const std::vector < std::vector<int>>& vector_of_vecors){
		int key_num = vector_of_vecors.size();
		mwSize *vector_len = new mwSize[key_num], key_num_size[1] = { key_num };
		for (int i = 0;i < key_num; i++)
			vector_len[i] = vector_of_vecors[i].size();

		mxArray *vector_of_vecors_cell = mxCreateCellArray(1, key_num_size);

		for (size_t i = 0; i < key_num; i++) {
			mxArray* blocksArr = mxCreateDoubleMatrix(1, vector_len[i], mxREAL);

			double* blocks_ptr = mxGetPr(blocksArr);
			for (int bi = 0; bi < vector_len[i]; bi++)
				blocks_ptr[bi] = vector_of_vecors[i][bi];

			mxSetCell(vector_of_vecors_cell, i, blocksArr);
		}
		return vector_of_vecors_cell;
	}

	mxArray* VectorOfSetsToCellArray(const std::vector < std::set<int>>& vector_of_vecors) {
		int key_num = vector_of_vecors.size();
		mwSize *vector_len = new mwSize[key_num], key_num_size[1] = { key_num };
		for (int i = 0;i < key_num; i++)
			vector_len[i] = vector_of_vecors[i].size();

		mxArray *vector_of_vecors_cell = mxCreateCellArray(1, key_num_size);

		for (size_t i = 0; i < key_num; i++) {
			mxArray* blocksArr = mxCreateDoubleMatrix(1, vector_len[i], mxREAL);

			double* blocks_ptr = mxGetPr(blocksArr);
			size_t bi = 0;
			for (auto set_element : vector_of_vecors[i]) {
				blocks_ptr[bi] = set_element;
				bi++;
			}

			mxSetCell(vector_of_vecors_cell, i, blocksArr);
		}
		return vector_of_vecors_cell;
	}

	bool FieldDataToPointer(const mxArray* struct_pnt,
		const char* field_name, double** out_data, int* M, int *N, int index) 	
	{
		const mxArray* field_array_ptr = mxGetField(struct_pnt, index, field_name);
		if (field_array_ptr == NULL) {
			return false;
			std::cout << "\nNo " << field_name << "field" << std::endl;
		}
		if (M) 	*M = mxGetM(field_array_ptr);
		if (N) 	*N = mxGetN(field_array_ptr);

		*out_data = mxGetPr(field_array_ptr);
		return true;
	}

	void DebugMex::Mat2Mfile(const Eigen::MatrixXd& M, const std::string& M_name) {
		if (!M_name.empty())
			m_file << "\n" << M_name << " = [";

		for (int i = 0; i < M.rows(); i++) {
			for (int j = 0; j < M.cols(); j++)
				m_file << M(i, j) << " ";
			m_file << ";";
		}
		if (!M_name.empty())
			m_file << " ];" << std::endl;
		PostOutput();
	}

	void DebugMex::MatInt2Mfile(const Eigen::MatrixXi& M, const std::string& M_name) {
		m_file << "\n" << M_name << " = [";
		for (int i = 0; i < M.rows(); i++) {
			for (int j = 0; j < M.cols(); j++)
				m_file << M(i, j) << " ";
			m_file << ";";
		}
		m_file << " ];" << std::endl;
		PostOutput();
	}


	void DebugMex::Mat2Mfile(const Eigen::VectorXd& V,
		const std::string& V_name,
		const std::vector<int>& index) {
		if (!V_name.empty())
			m_file << "\n" << V_name << " = [";
		if (index.empty())
			for (int i = 0; i < V.size(); i++) m_file << V(i) << " ";
		else
			for (auto i : index) m_file << V(i) << " ";

		if (!V_name.empty()) {
			m_file << " ];" << std::endl;
			PostOutput();
		}
	}
	void DebugMex::ColumnVec2Mat(const Eigen::VectorXd& V, 
								 const std::string& V_name,
								const std::vector<int>& index, int d) 
	{
		if (!V_name.empty())
			m_file << "\n" << V_name << " = [";

		if (index.empty()) {
			for (int ri = 0; ri < V.size()/d; ri++) {
				for (int ci = 0; ci < d; ci++)
					m_file << V(d*ri + ci) << " ";
				m_file << "; ";
			}
		} else {
			for (int ri : index) {
				for (int ci = 0; ci <d; ci++)
					m_file << V(d*ri + ci) << " ";
				m_file << "; ";
			}
		}

		if (!V_name.empty()) {
			m_file << " ];" << std::endl;
			PostOutput();
		}
	}

	void DebugMex::MatInt2Mfile(const Eigen::VectorXi& V, const std::string& V_name) {
		if (!V_name.empty())
			m_file << "\n" << V_name << " = [";

		for (int i = 0; i < V.size(); i++)
			m_file << V(i) << " ";
		if (!V_name.empty()) {
			m_file << " ];" << std::endl;
			PostOutput();
		}
	}

	void DebugMex::Mat2Mfile(const std::vector<Eigen::VectorXd>& V, const std::string& V_name) {
		m_file << "\n" << V_name << " = [";
		for (auto& Vrow : V) {
			Mat2Mfile(Vrow);
		}
		m_file << " ];" << std::endl;
		PostOutput();
	}


	void DebugMex::MatList2Mfile(const std::vector<Eigen::VectorXi>& V, const std::string& V_name) {
		m_file << "\n" << V_name << " = {";
		for (const auto& Vrow : V) {
			m_file << "[";
			MatInt2Mfile(Vrow);
			m_file << "] ";
		}
		m_file << " };" << std::endl;

		PostOutput();
	}


	void DebugMex::MatList2Mfile(const std::vector<std::vector<int>>& V,
		const std::string& V_name) {
		m_file << "\n" << V_name << " = {";
		for (const auto& Vrow : V) {
			m_file << "[" << Vrow << "] ;";
		}
		m_file << " };" << std::endl;

		PostOutput();
	}

	void DebugMex::MatList2Mfile(const std::vector<std::set<int>>& V,
		const std::string& V_name) {
		m_file << "\n" << V_name << " = {";
		for (const auto& Vrow : V) {
			m_file << "[" << Vrow << "] ;";
		}
		m_file << " };" << std::endl;

		PostOutput();
	}


	void DebugMex::MatList2Mfile(const std::vector<Eigen::Matrix2d>& V,
		const std::string& V_name
	) {
		m_file << "\n" << V_name << " = {";
		for (const auto& M : V) {
			m_file << "[";
			for (int i = 0; i < M.rows(); i++) {
				for (int j = 0; j < M.cols(); j++)	m_file << M(i, j) << " ";
				m_file << ";";
			}
			m_file << "],";
		}
		m_file << " };" << std::endl;
		PostOutput();
	}


	void DebugMex::MatList2Mfile(const std::vector<Eigen::Matrix2d>& V,
								const std::string& V_name,
								const std::vector<int>& index)
	{
		m_file << "\n" << V_name << " = {";
		for (auto i : index) {
			const auto& M = V[i];
				m_file << "[";
				for (int i = 0; i < M.rows(); i++) {
					for (int j = 0; j < M.cols(); j++)	m_file << M(i, j) << " ";
					m_file << ";";
				}
				m_file << "],";
		}
		m_file << " };" << std::endl;
		PostOutput();
	}

	void DebugMex::Mat2Mfile(const std::vector<Eigen::Vector2i>& V,
		const std::string& V_name,
		const std::vector<int>& index) {
		m_file << "\n" << V_name << " = [";
		if (index.empty())
			for (auto& Vrow : V) {
				MatInt2Mfile(Vrow);
				m_file << " ;";
			}
		else
			for (auto i : index) {
				MatInt2Mfile(V[i]);
				m_file << " ;";
			}

		m_file << " ];" << std::endl;
		PostOutput();
	}


	void DebugMex::Mat2Mfile(const std::vector<Eigen::Vector2d>& V,
		const std::string& V_name,
		const std::vector<int>& index) {
		m_file << "\n" << V_name << " = [";
		if (index.empty())
			for (auto& Vrow : V) {
				Mat2Mfile(Vrow);
				m_file << " ;";
			}
		else
			for (auto i : index) {
				Mat2Mfile(V[i]);
				m_file << " ;";
			}

		m_file << " ];" << std::endl;
		PostOutput();
	}
	void DebugMex::SparseMat2Mfile(const std::vector<Eigen::Triplet<double>>& M,
									int dim, const std::string& M_name) {
		m_file << "\n" << M_name << " = sparse([";
		for (auto m : M) 
			m_file << m.row()+1 << ",";
		
		m_file << "], [";
		for (auto m : M) {
			m_file << m.col()+1 << ",";
		}

		m_file << "], [";
		for (auto m : M) {
			m_file << m.value() << ",";
		}
		
		m_file <<"], "<< dim << "," << dim << " );";
		PostOutput();
	}

	void DebugMex::SetArraty2Mfile(const std::set<int>* V, int array_len, const std::string& V_name) {
		m_file << "\n" << V_name << " = [";
		for (int i = 0; i < array_len; i++)
			for (auto val : V[i])
				m_file << i << "," << val << ";";
		m_file << "]\n";

		PostOutput();
	}
	void DebugMex::SetArraty2Mfile(const std::vector<std::set<int>>& V, const std::string& V_name){
		m_file << "\n" << V_name << " = [";
		int array_len = V.size();
		for (int i = 0; i < array_len; i++)
			for (auto val : V[i])
				m_file << i << "," << val << ";";
		m_file << "]\n";

		PostOutput();
	}

	std::vector<int> DebugMex::ColumnStackIndex(const std::vector<int>& index, int d) {
		std::vector<int> flat_indices;

		for (auto i : index)
			for (int j = 0; j < d; j++)
				flat_indices.push_back(d*i + j);


		return flat_indices;
	}
} //endof namespace
