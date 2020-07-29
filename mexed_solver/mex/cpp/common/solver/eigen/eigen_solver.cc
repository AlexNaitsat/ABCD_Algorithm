// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il  (Alexander Naitsat)

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
}

void EigenSolver::UpdateMatrixEntryValue(
   const std::vector<EigenEntry>& entry_list) {
  system_.setFromTriplets(entry_list.begin(), entry_list.end());
}

void EigenSolver::AnalyzePattern() {
  system_.makeCompressed();

  system_solver_.analyzePattern(system_);
  is_pattern_analyzed = true;
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
								const std::vector<int>& fixed_vertices,
								int d ) {
	assert(lhs != nullptr);
	//const int d = 2;
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
}

}  // namespace eigen
}  // namespace solver
}  // namespace common
