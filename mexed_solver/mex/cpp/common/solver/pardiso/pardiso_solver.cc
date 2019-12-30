// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.ac.il (Alexander Naitsat)

#include "stdafx.h"
#include <omp.h>
#include "common/solver/pardiso/pardiso_solver.h"
#include "data_io/data_io_utils.h"
#include <algorithm>
#include <cassert>
#include <iostream>
namespace common
{
namespace solver
{
namespace pardiso
{

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
    printf("[PARDISO]: License check was successful ... \n");

  num_procs_ =  omp_get_max_threads();

  iparm_[2] = num_procs_;

  maxfct_ = 1;
  mnum_ = 1;

  msglvl_ = 0;
  error_ = 0;
}

void PardisoSolver::SetPattern(const std::vector<Eigen::Triplet<double>> &eigen_entry_list,
							  int dim)
{
	std::vector<MatrixEntry> entry_list;
	is_symmetric = (abs(mtype_) == 2);
	if (is_symmetric) {
		for (auto& eigen_entry : eigen_entry_list) {
			size_t col = eigen_entry.col(), row = eigen_entry.row();
			entry_list.emplace_back(MatrixEntry(row, col, eigen_entry.value()));
		}
	} else {
	entry_list.reserve(2*eigen_entry_list.size());
		for (auto& eigen_entry : eigen_entry_list) {
			size_t col = eigen_entry.col(), row = eigen_entry.row();
			entry_list.emplace_back(MatrixEntry(col, row, eigen_entry.value()) );
			if (row!=col )
				entry_list.emplace_back(MatrixEntry(row,col, eigen_entry.value()));
		}
	}

	SetPattern(entry_list,  dim);
}

void PardisoSolver::SetPatter4EigenUpper(std::vector<Eigen::Triplet<double>> &entry_eigen,
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
	#ifdef _CHECK_EIGEN_PRECISISON
		hessian_entry_list = entry_eigen;//copy to compare later solution precision  
	#endif

	std::sort(entry_eigen.begin(), entry_eigen.end(), [](Eigen::Triplet<double> a, Eigen::Triplet<double> b) {
		return a.row() == b.row() ? a.col() < b.col() : a.row() < b.row();
	});

	int nnz = 1;
	int pivot = -1;

	for (auto It = entry_eigen.begin(); It != entry_eigen.end(); It++) {
		double summed_value = It->value();
		while ((It + 1) != entry_eigen.end() && (It + 1)->col() == It->col() && (It + 1)->row() == It->row()) {
			summed_value += (It + 1)->value();
			It++;
		}
		ja_.push_back(It->col() + 1);
		a_.push_back(summed_value);
		if (It->row() != pivot)
		{
			ia_.push_back(nnz);
			pivot = It->row();
		}
		nnz++;
	}
	ia_.push_back(nnz);
}



void PardisoSolver::SetPattern(const std::vector<MatrixEntry> &entry_list,
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

  std::vector<MatrixEntry> entry = entry_list;
  std::sort(entry.begin(), entry.end(), [](MatrixEntry a, MatrixEntry b) {
    return a.row == b.row ? a.col < b.col : a.row < b.row;
  });

  int nnz = 1;
  int pivot = -1;

  std::vector<MatrixEntry> entry_unique;
  entry_unique.clear();

  for (auto It = entry.begin(); It != entry.end(); It++) {
	  double summed_value = It->val;
	  while ((It+1) != entry.end() && (It+1)->col == It->col  && (It+1)->row == It->row) {
		  summed_value += (It+1)->val;
		  It++;
      }
	  entry_unique.emplace_back(MatrixEntry(It->row, It->col, summed_value));
  }
  entry = entry_unique;


  for (const auto &item : entry)
  {
    ja_.push_back(item.col + 1);
    a_.push_back(item.val);
	coo_to_csr_map_[std::to_string(item.row) + "#" + std::to_string(item.col)] =
        a_.size() - 1;
    if (item.row != pivot)
    {
      ia_.push_back(nnz);
      pivot = item.row;
    }
    nnz++;
  }

  ia_.push_back(nnz);
}

void PardisoSolver::UpdateMatrixEntryValue(
    const std::vector<MatrixEntry> &entry_list)
{
  assert(entry_list.size() == a_.size());

  for (const auto &entry : entry_list)
  {
    const auto &item = coo_to_csr_map_.find(std::to_string(entry.row) + "#" +
                                            std::to_string(entry.col));
    assert(item != coo_to_csr_map_.end());
    a_[item->second] = entry.val;
  }
}

void PardisoSolver::AnalyzePattern()
{
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
}

bool PardisoSolver::Factorize()
{
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
										const std::vector<int>& fixed_vertices) {
	assert(lhs != nullptr);
	const int d = 2;
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

#ifdef _CHECK_EIGEN_PRECISISON
	int dim_ = num_rows_;
	Eigen::MatrixXd A(dim_, dim_);
	A.setZero();
	for (auto h_entry : hessian_entry_list) {
		A(h_entry.row(), h_entry.col()) += h_entry.value();
		if (h_entry.col() != h_entry.row())
			A(h_entry.col(), h_entry.row()) += h_entry.value();

	}

	Eigen::VectorXd  lhs_free_eigen( lhs_free.size()), 	rhs_free_eigen( rhs_free.size());
	for (int i = 0; i < lhs_free_eigen.size(); i++) {
			lhs_free_eigen[i] = lhs_free[i];
			rhs_free_eigen[i] = rhs_free[i];
	}

	double two_norm_diff = (A* lhs_free_eigen - rhs_free_eigen).norm();
	std::cout << "\n Precision of the solution=" << two_norm_diff << std::endl;
#endif 

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
