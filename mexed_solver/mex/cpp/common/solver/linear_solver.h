// Copyright @2019. All rights reserved.
// Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

//#define _USE_PARDISO 1 //uncomment for building Pardiso version
namespace common {
	namespace solver {
		class LinearSolver {
		public: 
			using EigenEntry = Eigen::Triplet<double>;
			virtual bool IsPatternAnalyzed() = 0;
			virtual bool IsInitialized() = 0;
			virtual void Init(int mtype) = 0;
			virtual void SetPattern(const std::vector<EigenEntry>& entry_list, int dim) = 0;
			virtual void UpdateMatrixEntryValue(const std::vector<EigenEntry>& entry_list)=0;
			virtual void AnalyzePattern() = 0;
			virtual bool Factorize() = 0;
			virtual void SolveDirichletConstraints(const Eigen::VectorXd& rhs,
													Eigen::VectorXd* result,
													const std::vector<int>& free_vertices,
													const std::vector<int>& fixed_vertices,
													int d=2) = 0;
			virtual void FreeSolver() = 0;
		};
	}
}