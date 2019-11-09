// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#include "stdafx.h"
#include "common/solver/eigen/eigen_solver.h"

#include <algorithm>
#include <cassert>

#include "data_io/data_io_utils.h"
namespace common {
namespace solver {
namespace eigen {

void EigenSolver::Init(int mtype) {}

void EigenSolver::SetPattern(const std::vector<EigenEntry>& entry_list,
                             int dim) {
  system_ = Eigen::SparseMatrix<double>(dim, dim);

  UpdateMatrixEntryValue(entry_list);

  dim_ = dim;
#ifdef _CHECK_EIGEN_PRECISISON
  hessian_entry_list = entry_list;
#endif
#ifdef _CHECK_PSD
  Eigen::MatrixXd A(dim_, dim_);
  A.setZero();
 
  for (auto h_entry : entry_list) {
	  A(h_entry.row(), h_entry.col()) += h_entry.value();
	  if (h_entry.col() != h_entry.row())
			A(h_entry.col(), h_entry.row()) += h_entry.value();
  }
  Eigen::LLT<Eigen::MatrixXd> lltOfA(A);
  if (lltOfA.info() == Eigen::NumericalIssue)
  {
	  Eigen::VectorXcd Hessian_eigens_complex = A.eigenvalues();
	  Eigen::VectorXd Hessian_eigens(Hessian_eigens_complex.rows());
	  std::cout << "\nNegative eigenvalues of the Hessian are:\n";
	  int num_of_negativ_eigens = 0;
	  const double eig_EPS = 1e-12;
	  for (int r = 0; r < Hessian_eigens.rows(); r++) {
		  Hessian_eigens(r) = Hessian_eigens_complex(r).real();
		  if (Hessian_eigens(r) < 0 && abs(Hessian_eigens(r)) > eig_EPS) {
			  std::cout << Hessian_eigens(r) << std::endl;
			  num_of_negativ_eigens++;
		  }
	  }
  }
#endif  
}

void EigenSolver::UpdateMatrixEntryValue(
    const std::vector<EigenEntry>& entry_list) {
  system_.setFromTriplets(entry_list.begin(), entry_list.end());
}

void EigenSolver::AnalyzePattern() {
  system_.makeCompressed();

  system_solver_.analyzePattern(system_);
}

bool EigenSolver::Factorize() {
  system_solver_.factorize(system_);
  return true;
}

void EigenSolver::Solve(const Eigen::VectorXd& rhs, Eigen::VectorXd* lhs) {
  assert(lhs != nullptr);

  *lhs = system_solver_.solve(rhs);
}

void EigenSolver::SolveDirichletConstraints(const Eigen::VectorXd& rhs,
								Eigen::VectorXd* lhs,
								const std::vector<int>& free_vertices,
								const std::vector<int>& fixed_vertices) {
	assert(lhs != nullptr);
	const int d = 2;
	size_t len = rhs.size();
	size_t free_ver_num = free_vertices.size();

	Eigen::VectorXd rhs_free(d*free_ver_num);
	for (int i = 0; i < free_ver_num; i++) {
		 int v = free_vertices[i];
		 for (int k=0; k < d; k++)
			rhs_free[d*i + k] = rhs[d*v + k];
	}

	 Eigen::VectorXd lhs_free;
	 Solve(rhs_free, &lhs_free);

	for (int i = 0; i < free_ver_num; i++) {
		int v = free_vertices[i];
		for (int k = 0; k < d; k++)
			(*lhs)[d*v+k] = lhs_free[d*i+k];
	}

	for (auto v : fixed_vertices)
		for (int k = 0; k < d; k++)
			(*lhs)[d*v + k] = 0;
	
	#ifdef _CHECK_EIGEN_PRECISISON
		Eigen::MatrixXd A(dim_, dim_);
		A.setZero();
		for (auto h_entry : hessian_entry_list) {
			A(h_entry.row(), h_entry.col()) += h_entry.value();
			if (h_entry.col() != h_entry.row())
				A(h_entry.col(), h_entry.row()) += h_entry.value();
			
		}
		double two_norm_diff = (A*lhs_free - rhs_free).norm();
		std::cout << "\n Precision of the solution=" << two_norm_diff << std::endl;
	#endif 
}

}  // namespace eigen
}  // namespace solver
}  // namespace common
