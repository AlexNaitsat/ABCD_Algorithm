// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace common {
namespace solver {
namespace eigen {

class EigenSolver {
 public:
  using EigenEntry = Eigen::Triplet<double>;

  EigenSolver(){};
  ~EigenSolver(){};

  void Init(int mtype);
  void SetPattern(const std::vector<EigenEntry>& entry_list, int dim);
  void SetPatternDirichletConstriantsKKT(std::vector<EigenEntry>& entry_list, 
									  const std::vector<EigenEntry>& constr_entry_list,
									  int dim,
									  int fixed_vert_num);

  void SetPatternDirichletConstriant(std::vector<EigenEntry>& entry_list,int dim);

  void UpdateMatrixEntryValue(const std::vector<EigenEntry>& entry_list);
  void AnalyzePattern();
  bool Factorize();
  void Solve(const Eigen::VectorXd& rhs, Eigen::VectorXd* result);
  void SolveDirichletConstraintsKKT(const Eigen::VectorXd& rhs, 
								 Eigen::VectorXd* result, 
								 int  fixed_ver_num);

  void SolveDirichletConstraints(const Eigen::VectorXd& rhs,
								Eigen::VectorXd* result,
								const std::vector<int>& free_vertices,
								const std::vector<int>& fixed_vertices);

  void FreeSolver(){};

 protected:
  Eigen::SparseMatrix<double> system_;

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation_;

  Eigen::SparseLU<Eigen::SparseMatrix<double>> lu_solver_;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>,Eigen::Upper> system_solver_;

  int dim_;
  std::vector<EigenEntry> hessian_entry_list;
};

}  // namespace eigen
}  // namespace solver
}  // namespace common
