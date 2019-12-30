// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.ac.il (Alexander Naitsat)
#include "stdafx.h"
#include "common/optimization/line_search_util.h"
#include <iostream>

#include "data_io/data_io_utils.h"

using namespace data_io;

namespace common
{
namespace optimization
{

#define TO_DOUBLE(x) ((double) x)

void ArmijoLineSearch(const std::vector<Eigen::Vector2d>& deformed,
                      const Eigen::VectorXd& search_direction,
                      mesh_distortion::IsotropicSVDEnergyModel2D& model,
                      std::vector<Eigen::Vector2d>* output) {
  assert(output != nullptr);

  *output = deformed;

  double fk = model.ComputeEnergy(*output);
  Eigen::VectorXd dfk;
  model.ComputeGradient(*output, &dfk);

  double step_size = 1.0;

  for (int i = 0; i < deformed.size(); i++) {
    (*output)[i][0] = deformed[i][0] + step_size * search_direction[2 * i + 0];
    (*output)[i][1] = deformed[i][1] + step_size * search_direction[2 * i + 1];
  }

  double dot_product = dfk.transpose() * search_direction;

  double lhs = model.ComputeEnergy(*output);
  double rhs = fk + model.ls_alpha * step_size * dot_product;
  while (lhs > rhs && step_size > 1e-10) {
    step_size *= model.ls_beta;

    for (int i = 0; i < deformed.size(); i++) {
      (*output)[i][0] =
          deformed[i][0] + step_size * search_direction[2 * i + 0];
      (*output)[i][1] =
          deformed[i][1] + step_size * search_direction[2 * i + 1];
    }

    lhs = model.ComputeEnergy(*output);
    rhs = fk + model.ls_alpha * step_size * dot_product;
  }
}

double ArmijoLineSearchInBlock(const std::vector<Eigen::Vector2d>& deformed,
							const Eigen::VectorXd& search_direction,
							mesh_distortion::IsotropicSVDEnergyModel2D& model,
							std::vector<Eigen::Vector2d>* output,
							const Eigen::VectorXd& dfk,
							double fk,
							const std::vector<int>& element_block,
							const std::vector<int>& free_vertex_block,
							const SolverSpecification& solverSpec) {
	assert(output != nullptr);

	double step_size = 1.0;
	
	double dot_product = 0; 
	for (auto i: free_vertex_block) {
		(*output)[i][0] = deformed[i][0] + step_size * search_direction[2 * i + 0];
		(*output)[i][1] = deformed[i][1] + step_size * search_direction[2 * i + 1];
		dot_product += dfk[2 * i + 0] * search_direction[2 * i + 0] +
					   dfk[2 * i + 1] * search_direction[2 * i + 1];
		
		if (mxIsNaN((*output)[i][0]) || mxIsNaN((*output)[i][1]))
		#pragma omp critical
		{
			std::cout << "\n Nan in LS-beggining output vertex=" << i;
		}
	}

	double lhs = model.ComputeEnergyInBlock(*output, element_block,solverSpec);

	double initial_energy = lhs;
	double rhs = fk + model.ls_alpha * step_size * dot_product;
	int ls_iter = 0;

	while (lhs > rhs && step_size > 1e-10) {
		step_size *= model.ls_beta;

		for (auto i : free_vertex_block) {
			(*output)[i][0] =
				deformed[i][0] + step_size * search_direction[2 * i + 0];
			(*output)[i][1] =
				deformed[i][1] + step_size * search_direction[2 * i + 1];
			if (mxIsNaN((*output)[i][0]) || mxIsNaN((*output)[i][1]))
			#pragma omp critical 
			{
				std::cout << "\n Nan in LS output vertex=" << i;
			}
		}

		lhs = model.ComputeEnergyInBlock(*output,element_block,solverSpec);
		rhs = fk + model.ls_alpha * step_size * dot_product;
		ls_iter++;
	}
		return lhs;
}



double ArmijoLineSearchEnhancedInBlock(const std::vector<Eigen::Vector2d>& deformed,
										const Eigen::VectorXd& search_direction,
										mesh_distortion::IsotropicSVDEnergyModel2D& model,
										std::vector<Eigen::Vector2d>* output,
										const Eigen::VectorXd& dfk, 
										double fk,
										const std::vector<int>& element_block,
										const std::vector<int>& free_vertex_block,
										const SolverSpecification& solverSpec) 
{
	assert(output != nullptr);

	double step_size = 1.0;

	double dot_product = 0;
	for (auto i : free_vertex_block) {
		(*output)[i][0] = deformed[i][0] + step_size * search_direction[2 * i + 0];
		(*output)[i][1] = deformed[i][1] + step_size * search_direction[2 * i + 1];
		dot_product += dfk[2 * i + 0] * search_direction[2 * i + 0] +
			dfk[2 * i + 1] * search_direction[2 * i + 1];

		if (mxIsNaN((*output)[i][0]) || mxIsNaN((*output)[i][1])) {
			#pragma omp critical
			{
				std::cout << "\n Nan in LS-beggining output vertex=" << i;
			}
		}
	}

	double lhs = model.ComputeEnergyInBlock(*output, element_block, solverSpec);

	double initial_energy = lhs;
	double best_step_size = 0, max_reduce_in_energy =0;

	double rhs = fk + model.ls_alpha * step_size * dot_product;
	int ls_iter = 0;

	while (lhs > rhs && step_size > 1e-16 && ls_iter < model.ls_max_iter) {
		step_size *= model.ls_beta;

		for (auto i : free_vertex_block) {
			(*output)[i][0] =
				deformed[i][0] + step_size * search_direction[2 * i + 0];
			(*output)[i][1] =
				deformed[i][1] + step_size * search_direction[2 * i + 1];
			if (mxIsNaN((*output)[i][0]) || mxIsNaN((*output)[i][1])) {
				#pragma omp critical 
				{
					std::cout << "\n Nan in LS output vertex=" << i;
				}
			}
		}

		lhs = model.ComputeEnergyInBlock(*output, element_block, solverSpec);
		rhs = fk + model.ls_alpha * step_size * dot_product;
		ls_iter++;
		if (initial_energy - lhs > max_reduce_in_energy) {
			max_reduce_in_energy = initial_energy - lhs;
			best_step_size = step_size;
		}

	}

	if (max_reduce_in_energy> 0 &&  max_reduce_in_energy > initial_energy - lhs) {
		lhs = initial_energy - max_reduce_in_energy;
		step_size = best_step_size;

		for (auto i : free_vertex_block) {
			(*output)[i][0] =
				deformed[i][0] + step_size * search_direction[2 * i + 0];
			(*output)[i][1] =
				deformed[i][1] + step_size * search_direction[2 * i + 1];
			if (mxIsNaN((*output)[i][0]) || mxIsNaN((*output)[i][1]))
				std::cout << "\n Nan in LS output vertex=" << i;
		}

	}
	return lhs;
}


double get_smallest_pos_quad_zero(double a, double b, double c) {
	double t1, t2;
	
	const double  polyCoefEps = 1e-16;
	const double  max_time = 2;
	if (abs(a) > polyCoefEps) {
		double delta_in = pow(b, 2) - 4 * a*c;
		if (delta_in < 0) {
			return INFINITY;
		}
		double delta = sqrt(delta_in);
		t1 = (-b + delta) / (2 * a);
		t2 = (-b - delta) / (2 * a);
	}
	else if (abs(b) > polyCoefEps){
		 t1 = t2 = -c / b;
	} else {
		return INFINITY; 
	}
	if (t1 < 0) t1 = max_time;
	if (t2 < 0) t2 = max_time;

	assert(std::isfinite(t1));
	assert(std::isfinite(t2));

	double tmp_n = std::min(t1, t2);
	t1 = std::max(t1, t2); t2 = tmp_n;
	if (t1 > 0) {
		if (t2 > 0) {
			return t2;
		}
		else {
			return t1;
		}
	}
	else {
		return INFINITY;
	}
}

double get_min_pos_root_2D(
	const std::vector<Eigen::Vector2d>& uv, const std::vector<Eigen::Vector3i>& F,
	const Eigen::VectorXd& d, int f) {
	int v1 = F[f](0); int v2 = F[f](1); int v3 = F[f](2);	
	#define U11 uv[v1](0)
	#define U12 uv[v1](1)
	#define U21 uv[v2](0)
	#define U22 uv[v2](1)
	#define U31 uv[v3](0)
	#define U32 uv[v3](1)

	#define V11 d(2*v1+0)
	#define V12 d(2*v1+1)
	#define V21 d(2*v2+0)
	#define V22 d(2*v2+1)
	#define V31 d(2*v3+0)
	#define V32 d(2*v3+1)

	double a = V11*V22 - V12*V21 - V11*V32 + V12*V31 + V21*V32 - V22*V31;
	double b = U11*V22 - U12*V21 - U21*V12 + U22*V11 - U11*V32 + U12*V31 + U31*V12 - U32*V11 + U21*V32 - U22*V31 - U31*V22 + U32*V21;
	double c = U11*U22 - U12*U21 - U11*U32 + U12*U31 + U21*U32 - U22*U31;
		
	return get_smallest_pos_quad_zero(a, b, c);
}

#define TwoPi  6.28318530717958648
const double eps = 1e-14;
int SolveP3(std::vector<double>& x, double a, double b, double c) {
	double a2 = a*a;
	double q = (a2 - 3 * b) / 9;
	double r = (a*(2 * a2 - 9 * b) + 27 * c) / 54;
	double r2 = r*r;
	double q3 = q*q*q;
	double A, B;
	if (r2<q3) {
		double t = r / sqrt(q3);
		if (t<-1) t = -1;
		if (t> 1) t = 1;
		t = acos(t);
		a /= 3; q = -2 * sqrt(q);
		x[0] = q*cos(t / 3) - a;
		x[1] = q*cos((t + TwoPi) / 3) - a;
		x[2] = q*cos((t - TwoPi) / 3) - a;
		return(3);
	}
	else {
		A = -pow(fabs(r) + sqrt(r2 - q3), 1. / 3);
		if (r<0) A = -A;
		B = A == 0 ? 0 : B = q / A;

		a /= 3;
		x[0] = (A + B) - a;
		x[1] = -0.5*(A + B) - a;
		x[2] = 0.5*sqrt(3.)*(A - B);
		if (fabs(x[2])<eps) { x[2] = x[1]; return(2); }
		return(1);
	}
}

Eigen::Vector3d compute_min_step_to_singularities(const std::vector<Eigen::Vector2d>& uv,
												  const std::vector<Eigen::Vector3i>& F,
												  const Eigen::VectorXd& d) {
	double min_time = INFINITY, max_time = 0;
	long double total_time = 0;
	int id, tri_num = F.size();

	for (int f = 0; f < tri_num; f++) {
		double min_positive_root = get_min_pos_root_2D(uv, F, d, f);
		min_time = std::min(min_time, min_positive_root);
		max_time = std::max(max_time, min_positive_root);
		total_time += min_positive_root;
	}

	return Eigen::Vector3d(min_time, max_time,total_time / TO_DOUBLE(tri_num));
}

Eigen::Vector3d compute_min_step_to_singularities_inBlock(const std::vector<Eigen::Vector2d>& uv,
														  const std::vector<Eigen::Vector3i>& F,
														  const Eigen::VectorXd& d,
														  const std::vector<int>& element_block,
														  mesh_distortion::IsotropicSVDEnergyModel2D& model) {
	double min_time = INFINITY, max_time = 0;
	long double total_time = 0;
	int id;
	Eigen::VectorXd flip_times(element_block.size());
	int i = 0;
	for (int f : element_block) {
		double min_positive_root;
		if (!(model.is_flip_barrier == model.is_element_valid[f]))
		{											
			min_positive_root = INFINITY;
		}
		else 
			min_positive_root = get_min_pos_root_2D(uv, F, d, f);
		flip_times(i) = std::min(min_positive_root, 2.0);i++;

		min_time = std::min(min_time, min_positive_root);
		max_time = std::max(max_time, min_positive_root);
		total_time += std::min(min_positive_root,2.0);   
	}

	return Eigen::Vector3d(min_time, max_time,
		total_time / TO_DOUBLE(element_block.size()) );

}

void ArmijoLineSearch3D(const std::vector<Eigen::Vector3d> &deformed,
						const Eigen::VectorXd &search_direction,
						mesh_distortion::IsotropicSVDEnergyModel3D &model,
						std::vector<Eigen::Vector3d> *output)
{
	assert(output != nullptr);

	*output = deformed;

	double fk = model.ComputeEnergy(*output);
	Eigen::VectorXd dfk;
	model.ComputeGradient(*output, &dfk);

	double ls_alpha = 0.2;
	double ls_beta = 0.8;
	double step_size = 1.0;

	for (int i = 0; i < deformed.size(); i++)
	{
		(*output)[i][0] = deformed[i][0] + step_size * search_direction[3 * i + 0];
		(*output)[i][1] = deformed[i][1] + step_size * search_direction[3 * i + 1];
		(*output)[i][2] = deformed[i][2] + step_size * search_direction[3 * i + 2];
	}

	double dot_product = dfk.transpose() * search_direction;

	double lhs = model.ComputeEnergy(*output);
	double rhs = fk + ls_alpha * step_size * dot_product;

	while (lhs > rhs && step_size > 1e-10)
	{
		step_size *= ls_beta;

		for (int i = 0; i < deformed.size(); i++)
		{
			(*output)[i][0] =
				deformed[i][0] + step_size * search_direction[3 * i + 0];
			(*output)[i][1] =
				deformed[i][1] + step_size * search_direction[3 * i + 1];
			(*output)[i][2] =
				deformed[i][2] + step_size * search_direction[3 * i + 2];
		}

		lhs = model.ComputeEnergy(*output);
		rhs = fk + ls_alpha * step_size * dot_product;
	}
}

double ArmijoLineSearchInBlock3D(const std::vector<Eigen::Vector3d> &deformed,
								 const Eigen::VectorXd &search_direction,
								 mesh_distortion::IsotropicSVDEnergyModel3D &model,
								 std::vector<Eigen::Vector3d> *output,
								 const Eigen::VectorXd &dfk, //gradient
								 double fk,
								 const std::vector<int> &element_block,
								 const std::vector<int> &free_vertex_block)
{
	assert(output != nullptr);

	double step_size = 1.0;

	double dot_product = 0;
	for (auto i : free_vertex_block)
	{
		(*output)[i][0] = deformed[i][0] + step_size * search_direction[3 * i + 0];
		(*output)[i][1] = deformed[i][1] + step_size * search_direction[3 * i + 1];
		(*output)[i][2] = deformed[i][2] + step_size * search_direction[3 * i + 2];
		dot_product += dfk[3 * i + 0] * search_direction[3 * i + 0] +
					   dfk[3 * i + 1] * search_direction[3 * i + 1] +
					   dfk[3 * i + 2] * search_direction[3 * i + 2];
	}
	double lhs = model.ComputeEnergyInBlock(*output, element_block);
	double initial_energy = lhs;
	double rhs = fk + model.ls_alpha * step_size * dot_product;
	int ls_iter = 0;

	while (lhs > rhs && step_size > 1e-10)
	{
		step_size *= model.ls_beta;

		for (auto i : free_vertex_block)
		{
			(*output)[i][0] =
				deformed[i][0] + step_size * search_direction[3 * i + 0];
			(*output)[i][1] =
				deformed[i][1] + step_size * search_direction[3 * i + 1];
			(*output)[i][2] =
				deformed[i][2] + step_size * search_direction[3 * i + 2];
		}

		lhs = model.ComputeEnergyInBlock(*output, element_block);
		rhs = fk + model.ls_alpha * step_size * dot_product;
		ls_iter++;
	}
	return lhs;
}

} // namespace optimization
} // namespace common
