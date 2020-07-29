// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il  (Alexander Naitsat)

#pragma once
#define _USE_PARDISO 1
#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Sparse>
#include "common/solver/linear_solver.h"
namespace common
{
namespace solver
{
namespace pardiso
{

extern "C"
{
  void pardisoinit(void *, int *, int *, int *, double *, int *);
  void pardiso(void *,
               int *,
               int *,
               int *,
               int *,
               int *,
               double *,
               int *,
               int *,
               int *,
               int *,
               int *,
               int *,
               double *,
               double *,
               int *,
               double *);
  void pardiso_chkmatrix(int *, int *, double *, int *, int *, int *);
  void pardiso_chkvec(int *, int *, double *, int *);
  void pardiso_printstats(int *, int *, double *, int *, int *, int *, double *, int *);
}

class PardisoSolver : public  LinearSolver {
public:
  struct MatrixEntry
  {
    int row;
    int col;
    double val;

    MatrixEntry(int r, int c, double v) : row(r), col(c), val(v) {}
  };
  
  //from base class
  bool IsPatternAnalyzed() {
	  return is_pattern_analyzed;
  }

  bool IsInitialized() { return is_initialized; }

  bool is_symmetric = false;
  PardisoSolver(){};
  ~PardisoSolver();

  void Init(int mtype);
  void SetPattern(std::vector<MatrixEntry> &entry_list, int dim);
  
  void SetPattern(const std::vector<EigenEntry> &eigen_entry_list, int dim);

  void UpdateMatrixEntryValue(const std::vector<MatrixEntry> &entry_list);
  void UpdateMatrixEntryValue(const std::vector<Eigen::Triplet<double>> &entry_list);//compatible with eigen

  void AnalyzePattern();
  bool Factorize();
  void Solve(const std::vector<double> &rhs, std::vector<double> *result);
  void PardisoSolver::SolveDirichletConstraints(const Eigen::VectorXd& rhs,
											    Eigen::VectorXd* lhs,
											    const std::vector<int>& free_vertices,
											    const std::vector<int>& fixed_vertices,
												int d =2);

  void FreeSolver();

protected:
  int num_rows_;
  bool is_pattern_analyzed = false;

  std::vector<int> ia_;
  std::vector<int> ja_;
  std::vector<double> a_;
  std::unordered_map<std::string, int> coo_to_csr_map_;
  std::vector<int> entry2a_, same_entries;
  std::vector<std::pair<Eigen::Triplet<double>, int> > entires_indices;

  int mtype_;
  int nrhs_ = 1;
  void *pt_[64];
  int iparm_[64];
  double dparm_[64];
  int maxfct_, mnum_, phase_, error_, msglvl_, solver_ = 0;
  int num_procs_;
  char *var_;
  double ddum_;
  int idum_;
  std::vector<Eigen::Triplet<double>> hessian_entry_list;
  bool is_initialized = false;
};

} // namespace pardiso
} // namespace solver
} // namespace common
