// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#include <iostream>

#include "common/solver/pardiso/pardiso_solver.h"

int main(int argc, const char *argv[])
{
  std::vector<common::solver::pardiso::PardisoSolver::MatrixEntry> entry_list;

  entry_list.emplace_back(0, 0, 0.1516);
  entry_list.emplace_back(0, 2, 0.3222);
  entry_list.emplace_back(2, 0, 0.3222);
  entry_list.emplace_back(1, 1, 0.0068);
  entry_list.emplace_back(3, 3, 0.4694);
  entry_list.emplace_back(2, 3, 0.4466);
  entry_list.emplace_back(3, 2, 0.4466);
  entry_list.emplace_back(2, 2, 0.2339);

  std::vector<double> rhs(4);
  rhs[0] = 0.2974;
  rhs[1] = 0.4137;
  rhs[2] = 0.0119;
  rhs[3] = 0.8255;

  std::vector<double> res(4);

  common::solver::pardiso::PardisoSolver m_solver;

  m_solver.Init(11);
  m_solver.SetPattern(entry_list, 4);
  m_solver.AnalyzePattern();
  m_solver.Factorize();
  m_solver.Solve(rhs, &res);

  for (int i = 0; i < res.size(); i++)
  {
    std::cout << res[i] << std::endl;
  }
  std::cout << std::endl;

  entry_list.clear();
  entry_list.emplace_back(0, 0, 3.1516);
  entry_list.emplace_back(0, 2, 1.3222);
  entry_list.emplace_back(2, 0, 1.3222);
  entry_list.emplace_back(1, 1, 2.0068);
  entry_list.emplace_back(3, 3, 4.4694);
  entry_list.emplace_back(2, 3, 3.4466);
  entry_list.emplace_back(3, 2, 3.4466);
  entry_list.emplace_back(2, 2, 1.2339);

  m_solver.UpdateMatrixEntryValue(entry_list);
  m_solver.Factorize();
  m_solver.Solve(rhs, &res);

  for (int i = 0; i < res.size(); i++)
  {
    std::cout << res[i] << std::endl;
  }
  std::cout << std::endl;

  m_solver.FreeSolver();

  return 0;
}
