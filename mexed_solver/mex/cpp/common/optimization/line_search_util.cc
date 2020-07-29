// Copyright @2019. All rights reserved.
// Authors: anaitsat@campus.technion.ac.il (Alexander Naitsat)
// 			mike323zyf@gmail.com (Yufeng Zhu)
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

double ArmijoLineSearchEnhancedInBlock3D(const std::vector<Eigen::Vector3d>& deformed,
										const Eigen::VectorXd& search_direction,
										mesh_distortion::IsotropicSVDEnergyModel3D& model,
										std::vector<Eigen::Vector3d>* output,
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
		(*output)[i][0] = deformed[i][0] + step_size * search_direction[3 * i + 0];
		(*output)[i][1] = deformed[i][1] + step_size * search_direction[3 * i + 1];
		(*output)[i][2] = deformed[i][2] + step_size * search_direction[3 * i + 2];
		dot_product += dfk[3 * i + 0] * search_direction[3 * i + 0] +
					   dfk[3 * i + 1] * search_direction[3 * i + 1] +
					   dfk[3 * i + 2] * search_direction[3 * i + 2] ;


		if (mxIsNaN((*output)[i][0]) || mxIsNaN((*output)[i][1]) || mxIsNaN((*output)[i][2])) {
#pragma omp critical
			{
				std::cout << "\n Nan in LS-beggining output vertex=" << i;
				mexErrMsgTxt("NAN value is found");
			}
		}
	}

	double lhs = model.ComputeEnergyInBlock(*output, element_block, solverSpec);

	double initial_energy = fk;
	double best_step_size = 0, max_reduce_in_energy = 0;

	double rhs = fk + model.ls_alpha * step_size * dot_product;
	int ls_iter = 0;

	while (lhs > rhs && step_size > 1e-16 && ls_iter < model.ls_max_iter) {
		step_size *= model.ls_beta;

		for (auto i : free_vertex_block) {
			for (int j = 0; j < 3; j++) 
				(*output)[i][j] = deformed[i][j] + step_size * search_direction[3 * i + j];
			

			if (mxIsNaN((*output)[i][0]) || mxIsNaN((*output)[i][1]) || mxIsNaN((*output)[i][2])) {
			#pragma omp critical 
				{
					std::cout << "\n Nan in LS output vertex=" << i;
					mexErrMsgTxt("NAN value is found");
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

	if (max_reduce_in_energy> 0 && max_reduce_in_energy > initial_energy - lhs) {
		lhs = initial_energy - max_reduce_in_energy;
		step_size = best_step_size;

		for (auto i : free_vertex_block) {
			for (int j = 0; j < 3; j++) 
				(*output)[i][j] = deformed[i][j] + step_size * search_direction[3 * i + j];
			
			if (mxIsNaN((*output)[i][0]) || mxIsNaN((*output)[i][1]) || mxIsNaN((*output)[i][2]))
#pragma omp critical 
			{
				std::cout << "\n Nan in LS output vertex=" << i;
				mexErrMsgTxt("NAN value is found");
			}
		}

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
				mexErrMsgTxt("NAN value is found");
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
					mexErrMsgTxt("NAN value is found");
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
			if (mxIsNaN((*output)[i][0]) || mxIsNaN((*output)[i][1])) {
				#pragma omp critical
					std::cout << "\n Nan in LS output vertex=" << i;
					mexErrMsgTxt("NAN value is found");
			}
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


double ArmijoLineSearchInBlock3D(const std::vector<Eigen::Vector3d> &deformed,
								 const Eigen::VectorXd &search_direction,
								 mesh_distortion::IsotropicSVDEnergyModel3D &model,
								 std::vector<Eigen::Vector3d> *output,
								 const Eigen::VectorXd &dfk, //gradient
								 double fk,
								 const std::vector<int> &element_block,
								 const std::vector<int> &free_vertex_block,
								 const data_io::SolverSpecification& solverSpec)

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
	double lhs = model.ComputeEnergyInBlock(*output, element_block,solverSpec);
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

		lhs = model.ComputeEnergyInBlock(*output, element_block, solverSpec);
		rhs = fk + model.ls_alpha * step_size * dot_product;
		ls_iter++;
	}
	return lhs;
}


// =========== 3D line search filtering =================

double getSmallestPositiveRealCubicRoot(double a, double b, double c, double d, double tol)
{
	using namespace std;

	// return negative value if no positive real root is found
	double t = -1, t_draft=-1;

	if (abs(a) <= tol) {
		//t = getSmallestPositiveRealQuadRoot(b, c, d, tol);
		t = get_smallest_pos_quad_zero(b, c, d);
	}
	else
	{
		complex<double> i(0, 1);
		complex<double> delta0(b*b - 3 * a*c, 0);
		complex<double> delta1(2 * b*b*b - 9 * a*b*c + 27 * a*a*d, 0);
		complex<double> C = pow((delta1 + sqrt(delta1*delta1 - 4.0 * delta0*delta0*delta0)) / 2.0, 1.0 / 3.0);

		complex<double> u2 = (-1.0 + sqrt(3.0)*i) / 2.0;
		complex<double> u3 = (-1.0 - sqrt(3.0)*i) / 2.0;

		complex<double> t1 = (b + C + delta0 / C) / (-3.0*a);
		complex<double> t2 = (b + u2*C + delta0 / (u2*C)) / (-3.0*a);
		complex<double> t3 = (b + u3*C + delta0 / (u3*C)) / (-3.0*a);
		
		const double  max_time = 2;
		double t1_ = max_time, t2_ = max_time, t3_ = max_time;

		if ((abs(imag(t1)) < tol) && (real(t1) > 0)) {
			t1_ = real(t1);
			//t = real(t1);
		}
		if ((abs(imag(t2)) < tol) && (real(t2) > 0) && ((real(t2) < t) || (t < 0))) {
			t2_ = real(t2);
			//t = real(t2);
		}
		if ((abs(imag(t3)) < tol) && (real(t3) > 0) && ((real(t3) < t) || (t < 0))) {
			t3_ = real(t3);
			//t = real(t3);
		}
		t = std::min( std::min(t1_, t2_), t3_);
	}
	return t;
}


// F-tet vertex indixes,  x - target vertex coordinates, p - descent direction
// f- tet index 
double  get_min_pos_root_3D(const  std::vector<Eigen::Vector3d>&x,  std::vector<Eigen::Vector4i> F,
								   const Eigen::VectorXd& p, int f)
{
	double tol = 1e-9, output = -1.0;
	//int n_tri = F.rows();
	double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
	double p1, p2, p3, p4, q1, q2, q3, q4, r1, r2, r3, r4;
	double a, b, c, d, t;

	int v1 = F[f](0), v2 = F[f](1), v3 = F[f](2), v4 = F[f](3);

		x1 = x[v1](0); //(xi,yi,zi) - target coordinates for i_th vertex of current tet.
		x2 = x[v2](0);
		x3 = x[v3](0);
		x4 = x[v4](0);

		y1 = x[v1](1);
		y2 = x[v2](1);
		y3 = x[v3](1);
		y4 = x[v4](1);

		z1 = x[v1](2);
		z2 = x[v2](2);
		z3 = x[v3](2);
		z4 = x[v4](2);

		p1 = p(3*v1+0); //(pi,qi,ri) - descent direction for i_th vertex of current tet.
		p2 = p(3*v2+0);
		p3 = p(3*v3+0);
		p4 = p(3*v4+0);

		q1 = p(3*v1+1);
		q2 = p(3*v2+1);
		q3 = p(3*v3+1);
		q4 = p(3*v4+1);

		r1 = p(3*v1+2);
		r2 = p(3*v2+2);
		r3 = p(3*v3+2);
		r4 = p(3*v4+2);

		a = -p1*q2*r3 + p1*r2*q3 + q1*p2*r3 - q1*r2*p3 - r1*p2*q3 + r1*q2*p3 + p1*q2*r4 - p1*r2*q4 - q1*p2*r4 + q1*r2*p4 + r1*p2*q4 - r1*q2*p4 - p1*q3*r4 + p1*r3*q4 + q1*p3*r4 - q1*r3*p4 - r1*p3*q4 + r1*q3*p4 + p2*q3*r4 - p2*r3*q4 - q2*p3*r4 + q2*r3*p4 + r2*p3*q4 - r2*q3*p4;
		b = -x1*q2*r3 + x1*r2*q3 + y1*p2*r3 - y1*r2*p3 - z1*p2*q3 + z1*q2*p3 + x2*q1*r3 - x2*r1*q3 - y2*p1*r3 + y2*r1*p3 + z2*p1*q3 - z2*q1*p3 - x3*q1*r2 + x3*r1*q2 + y3*p1*r2 - y3*r1*p2 - z3*p1*q2 + z3*q1*p2 + x1*q2*r4 - x1*r2*q4 - y1*p2*r4 + y1*r2*p4 + z1*p2*q4 - z1*q2*p4 - x2*q1*r4 + x2*r1*q4 + y2*p1*r4 - y2*r1*p4 - z2*p1*q4 + z2*q1*p4 + x4*q1*r2 - x4*r1*q2 - y4*p1*r2 + y4*r1*p2 + z4*p1*q2 - z4*q1*p2 - x1*q3*r4 + x1*r3*q4 + y1*p3*r4 - y1*r3*p4 - z1*p3*q4 + z1*q3*p4 + x3*q1*r4 - x3*r1*q4 - y3*p1*r4 + y3*r1*p4 + z3*p1*q4 - z3*q1*p4 - x4*q1*r3 + x4*r1*q3 + y4*p1*r3 - y4*r1*p3 - z4*p1*q3 + z4*q1*p3 + x2*q3*r4 - x2*r3*q4 - y2*p3*r4 + y2*r3*p4 + z2*p3*q4 - z2*q3*p4 - x3*q2*r4 + x3*r2*q4 + y3*p2*r4 - y3*r2*p4 - z3*p2*q4 + z3*q2*p4 + x4*q2*r3 - x4*r2*q3 - y4*p2*r3 + y4*r2*p3 + z4*p2*q3 - z4*q2*p3;
		c = -x1*y2*r3 + x1*z2*q3 + x1*y3*r2 - x1*z3*q2 + y1*x2*r3 - y1*z2*p3 - y1*x3*r2 + y1*z3*p2 - z1*x2*q3 + z1*y2*p3 + z1*x3*q2 - z1*y3*p2 - x2*y3*r1 + x2*z3*q1 + y2*x3*r1 - y2*z3*p1 - z2*x3*q1 + z2*y3*p1 + x1*y2*r4 - x1*z2*q4 - x1*y4*r2 + x1*z4*q2 - y1*x2*r4 + y1*z2*p4 + y1*x4*r2 - y1*z4*p2 + z1*x2*q4 - z1*y2*p4 - z1*x4*q2 + z1*y4*p2 + x2*y4*r1 - x2*z4*q1 - y2*x4*r1 + y2*z4*p1 + z2*x4*q1 - z2*y4*p1 - x1*y3*r4 + x1*z3*q4 + x1*y4*r3 - x1*z4*q3 + y1*x3*r4 - y1*z3*p4 - y1*x4*r3 + y1*z4*p3 - z1*x3*q4 + z1*y3*p4 + z1*x4*q3 - z1*y4*p3 - x3*y4*r1 + x3*z4*q1 + y3*x4*r1 - y3*z4*p1 - z3*x4*q1 + z3*y4*p1 + x2*y3*r4 - x2*z3*q4 - x2*y4*r3 + x2*z4*q3 - y2*x3*r4 + y2*z3*p4 + y2*x4*r3 - y2*z4*p3 + z2*x3*q4 - z2*y3*p4 - z2*x4*q3 + z2*y4*p3 + x3*y4*r2 - x3*z4*q2 - y3*x4*r2 + y3*z4*p2 + z3*x4*q2 - z3*y4*p2;
		d = x1*z2*y3 - x1*y2*z3 + y1*x2*z3 - y1*z2*x3 - z1*x2*y3 + z1*y2*x3 + x1*y2*z4 - x1*z2*y4 - y1*x2*z4 + y1*z2*x4 + z1*x2*y4 - z1*y2*x4 - x1*y3*z4 + x1*z3*y4 + y1*x3*z4 - y1*z3*x4 - z1*x3*y4 + z1*y3*x4 + x2*y3*z4 - x2*z3*y4 - y2*x3*z4 + y2*z3*x4 + z2*x3*y4 - z2*y3*x4;


		t = getSmallestPositiveRealCubicRoot(a, b, c, d, tol);

		if (t >= 0)
		{
			output = t;
		}
		else {
			output = 1e20;
		}
	//}
		return output;
}


Eigen::Vector3d compute_min_step_to_singularities_3d_inBlock(const std::vector<Eigen::Vector3d>& uv,
															const std::vector<Eigen::Vector4i>& F,
															const Eigen::VectorXd& d,
															const std::vector<int>& element_block,
															mesh_distortion::IsotropicSVDEnergyModel3D& model) {
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
			min_positive_root = get_min_pos_root_3D(uv, F, d, f);
		flip_times(i) = std::min(min_positive_root, 2.0);i++;

		min_time = std::min(min_time, min_positive_root);
		max_time = std::max(max_time, min_positive_root);
		total_time += std::min(min_positive_root, 2.0);
	}

	return Eigen::Vector3d(min_time, max_time,
		total_time / TO_DOUBLE(element_block.size()));

}

} // namespace optimization
} // namespace common
