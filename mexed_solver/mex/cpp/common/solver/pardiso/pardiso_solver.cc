// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il  (Alexander Naitsat)

#include "stdafx.h"
#include <omp.h>
#include "common/solver/pardiso/pardiso_solver.h"
#include "data_io/data_io_utils.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <time.h>
namespace common
{
namespace solver
{
namespace pardiso
{
#ifndef _OPENMP
	int omp_get_max_threads() { return 4; }
#endif 



void PardisoSolver::Init(int mtype)
{
  mtype_ = mtype;

  if (mtype_ == -1)
  {
    printf("Pardiso mtype not set.");
    exit(1);
  }

  error_ = 0;
  solver_ = 0;
  pardisoinit(pt_, &mtype_, &solver_, iparm_, dparm_, &error_);

  if (error_ != 0)
  {
    if (error_ == -10)
      printf("No license file found \n");
    if (error_ == -11)
      printf("License is expired \n");
    if (error_ == -12)
      printf("Wrong username or hostname \n");
    exit(1);
  }
  else
    //printf("[PARDISO]: License check was successful ... \n");

  num_procs_ =  omp_get_max_threads();

  iparm_[2] = num_procs_;

  maxfct_ = 1;
  mnum_ = 1;

  msglvl_ = 0;
  error_ = 0;

  is_initialized = true;
}
void PardisoSolver::SetPattern(const std::vector<Eigen::Triplet<double>> &entry_eigen,
		int dim)
{
	ia_.clear();
	ja_.clear();
	a_.clear();
	coo_to_csr_map_.clear();

	if (mtype_ == -1)
	{
		printf("Pardiso mtype not set.");
		exit(1);
	}

	num_rows_ = dim;
	entires_indices.clear();
	entires_indices.reserve(entry_eigen.size());
	for (auto It = entry_eigen.begin(); It != entry_eigen.end(); It++) {
		entires_indices.push_back(std::make_pair(*It, It - entry_eigen.begin()));
	}

	std::sort(entires_indices.begin(), entires_indices.end(), [](std::pair<Eigen::Triplet<double>, int> a,
		std::pair<Eigen::Triplet<double>, int> b) {
		return a.first.row() == b.first.row() ? a.first.col() < b.first.col() : a.first.row() < b.first.row();
	});

	int nnz = 1;
	int pivot = -1;
	entry2a_.clear();
	same_entries.clear();
	int It_i = 0;
	for (auto It = entires_indices.begin(); It != entires_indices.end(); It++) {
		double summed_value = It->first.value();
		entry2a_.push_back(a_.size());
		same_entries.push_back(1);
		while ((It + 1) != entires_indices.end() && 
			    (It + 1)->first.col() == It->first.col() && 
			    (It + 1)->first.row() == It->first.row()) 
		{
			summed_value += (It + 1)->first.value();
			It++;
			same_entries.back()++;
		}
		ja_.push_back(It->first.col() + 1);
		a_.push_back(summed_value); 
		if (It->first.row() != pivot)
		{
			ia_.push_back(nnz);
			pivot = It->first.row();
		}
		nnz++;
	}

	ia_.push_back(nnz);
}

void PardisoSolver::UpdateMatrixEntryValue( const std::vector<EigenEntry> &entry_list)
{
	int entry_i = 0;
	a_.assign(a_.size(), 0);
	int repetition_num = same_entries[entry_i];
	//for (const auto& entry : entry_list) {
	for (const auto& entry_index : entires_indices) {
		if (!repetition_num) {
			entry_i++;
			repetition_num = same_entries[entry_i];
		}
		a_[entry2a_[entry_i]] += entry_list[entry_index.second].value();
		repetition_num--;
	}
}

void PardisoSolver::AnalyzePattern()
{
	//std::cout << "\n Pardiso AnalyzePattern";
  if (mtype_ == -1)
  {
    printf("Pardiso mtype not set.");
    exit(1);
  }

  phase_ = 11;

  pardiso(pt_,
          &maxfct_,
          &mnum_,
          &mtype_,
          &phase_,
          &num_rows_,
          a_.data(),
          ia_.data(),
          ja_.data(),
          &idum_,
          &nrhs_,
          iparm_,
          &msglvl_,
          &ddum_,
          &ddum_,
          &error_,
          dparm_);

  if (error_ != 0)
  {
    printf("\nERROR during symbolic factorization: %d", error_);
    exit(1);
  }
  is_pattern_analyzed = true;
}

bool PardisoSolver::Factorize()
{
  //std::cout << "\n Pardiso Factorize";
  if (mtype_ == -1)
  {
    printf("Pardiso mtype not set.");
    exit(1);
  }

  phase_ = 22;

  pardiso(pt_,
          &maxfct_,
          &mnum_,
          &mtype_,
          &phase_,
          &num_rows_,
          a_.data(),
          ia_.data(),
          ja_.data(),
          &idum_,
          &nrhs_,
          iparm_,
          &msglvl_,
          &ddum_,
          &ddum_,
          &error_,
          dparm_);

  if (error_ != 0)
  {
    printf("\nERROR during numerical factorization: %d", error_);
    exit(2);
  }

  return (error_ == 0);
}

void PardisoSolver::Solve(const std::vector<double> &input,
                          std::vector<double> *result)
{
  assert(result != nullptr);
  std::vector<double> rhs = input;

  if (mtype_ == -1)
  {
    printf("Pardiso mtype not set.");
    exit(1);
  }

  result->resize(num_rows_);
 phase_ = 33;


  iparm_[7] = 1;

  pardiso(pt_,
          &maxfct_,
          &mnum_,
          &mtype_,
          &phase_,
          &num_rows_,
          a_.data(),
          ia_.data(),
          ja_.data(),
          &idum_,
          &nrhs_,
          iparm_,
          &msglvl_,
          rhs.data(),
          result->data(),
          &error_,
          dparm_);

  if (error_ != 0)
  {
    printf("\nERROR during solution: %d", error_);
    exit(3);
  }
}

void PardisoSolver::SolveDirichletConstraints(const Eigen::VectorXd& rhs,
											Eigen::VectorXd* lhs,
											const std::vector<int>& free_vertices,
											const std::vector<int>& fixed_vertices,
											int d) {
	assert(lhs != nullptr);
	size_t len = rhs.size();
	size_t free_ver_num = free_vertices.size();

	std::vector<double> rhs_free(d*free_ver_num);
	for (int i = 0; i < free_ver_num; i++) {
		int v = free_vertices[i];
		for (int k = 0; k < d; k++)
			rhs_free[d*i + k] = rhs[d*v + k];
	}

	std::vector<double> lhs_free;
	Solve(rhs_free, &lhs_free);

	for (int i = 0; i < free_ver_num; i++) {
		int v = free_vertices[i];
		for (int k = 0; k < d; k++)
			(*lhs)[d*v + k] = lhs_free[d*i + k];
	}

	for (auto v : fixed_vertices)
		for (int k = 0; k < d; k++)
			(*lhs)[d*v + k] = 0;

}

void PardisoSolver::FreeSolver()
{
  phase_ = 0;

  pardiso(pt_,
          &maxfct_,
          &mnum_,
          &mtype_,
          &phase_,
          &num_rows_,
          &ddum_,
          ia_.data(),
          ja_.data(),
          &idum_,
          &nrhs_,
          iparm_,
          &msglvl_,
          &ddum_,
          &ddum_,
          &error_,
          dparm_);
}

PardisoSolver::~PardisoSolver()
{
  if (mtype_ == -1)
    return;

  phase_ = -1;

  pardiso(pt_,
          &maxfct_,
          &mnum_,
          &mtype_,
          &phase_,
          &num_rows_,
          &ddum_,
          ia_.data(),
          ja_.data(),
          &idum_,
          &nrhs_,
          iparm_,
          &msglvl_,
          &ddum_,
          &ddum_,
          &error_,
          dparm_);
}

} // namespace pardiso
} // namespace solver
} // namespace common
