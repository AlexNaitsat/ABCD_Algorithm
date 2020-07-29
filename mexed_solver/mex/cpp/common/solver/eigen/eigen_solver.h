// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il  (Alexander Naitsat)

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "common/solver/linear_solver.h"
namespace common {
namespace solver {
namespace eigen {

class EigenSolver : public  LinearSolver {
 public:
  using EigenEntry = Eigen::Triplet<double>;

  EigenSolver(){};
  ~EigenSolver(){};

  bool IsPatternAnalyzed() {
	  return is_pattern_analyzed;

  }

  void Init(int mtype);
  bool IsInitialized() { return true; };
  void SetPattern(const std::vector<EigenEntry>& entry_list, int dim);
  void AnalyzePattern();
  bool Factorize();
  void FreeSolver() {};

  
  void SetPatternDirichletConstriantsKKT(std::vector<EigenEntry>& entry_list, 
									  const std::vector<EigenEntry>& constr_entry_list,
									  int dim,
									  int fixed_vert_num);

  void SetPatternDirichletConstriant(std::vector<EigenEntry>& entry_list,int dim);
  

  void UpdateMatrixEntryValue(const  std::vector<EigenEntry>& entry_list);
  void Solve(const Eigen::VectorXd& rhs, Eigen::VectorXd* result);
  void SolveDirichletConstraintsKKT(const Eigen::VectorXd& rhs, 
								 Eigen::VectorXd* result, 
								 int  fixed_ver_num);

  void SolveDirichletConstraints(const Eigen::VectorXd& rhs,
								Eigen::VectorXd* result,
								const std::vector<int>& free_vertices,
								const std::vector<int>& fixed_vertices,
								int d=2);


 protected:
  Eigen::SparseMatrix<double> system_;

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation_;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>,Eigen::Upper> system_solver_;


  int dim_;
  bool is_pattern_analyzed = false;
  std::vector<EigenEntry> hessian_entry_list;
};

}  // namespace eigen
}  // namespace solver
}  // namespace common
