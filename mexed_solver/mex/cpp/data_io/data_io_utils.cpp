// Copyright @2019. All rights reserved.
// Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

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
			mxArray* blocksArr = mxCreateDoubleMatrix(vector_len[i],1, mxREAL);

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


} //endof namespace
