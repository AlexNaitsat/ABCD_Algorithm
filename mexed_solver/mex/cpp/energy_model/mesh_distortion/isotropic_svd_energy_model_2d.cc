// Copyright @2019. All rights reserved.
// Authors: anaitsat@campus.technion.ac.il (Alexander Naitsat)
//          mike323zyf@gmail.com (Yufeng Zhu)

#include "stdafx.h"
#include "energy_model/distortion_kernel/arap_kernel_2d.h"

#include "energy_model/mesh_distortion/isotropic_svd_energy_model_2d.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <utility>
#include <iostream>

#include "common/util/linalg_util.h"
#include "data_io/data_io_utils.h"

namespace mesh_distortion {

namespace {

double ComputeTriangleVolume(const Eigen::Matrix2d& mat) {
  double volume = mat.determinant() * 0.5;
  return volume;
}

double ComputeTriangleVolume(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2)
{
	Eigen::Vector3d u = p1 - p0;
	Eigen::Vector3d v = p2 - p0;
	double volume = 0.5 * sqrt(u.dot(u) * v.dot(v) - u.dot(v) * u.dot(v));
	return volume;
}

}  // namespace

void IsotropicSVDEnergyModel2D::SetRestMesh( const Eigen::MatrixXd& position,
											const std::vector<Eigen::Vector3i>& mesh)
{
  size_t num_of_element = mesh.size();
  mesh_ = mesh;
  volume_.resize(num_of_element);
  Eigen::Matrix2d Eye2D;
  Eye2D << 1, 0, 0, 1;

  is_element_valid.resize(num_of_element, true);
  inverse_material_space_.resize(num_of_element);
  deformation_gradient_differential_.resize(4 * num_of_element, Eye2D);
  
  ut_df_v.resize(4 * num_of_element, Eye2D);
  svd_u_.resize(num_of_element, Eye2D);
  svd_v_.resize(num_of_element, Eye2D);
  svd_s_.resize(num_of_element,Eigen::Vector2d(1.0,1.0));

  element_distortion.resize(num_of_element);
  element_distortion.setZero();
  
  const Eigen::Vector2d kOO(0.0, 0.0);
  const Eigen::Vector2d kFO(1.0, 0.0);
  const Eigen::Vector2d kSO(0.0, 1.0);
  bool is_uv_mesh = (position.cols() == 3);

  total_volume = 0;
  for (size_t i = 0; i < num_of_element; i++) {
    Eigen::Vector3i triangle = mesh[i];
	
	Eigen::Matrix2d material_space;
	if (is_uv_mesh) {
		Eigen::Vector3d p0, p1, p2;
		
		util::MatrixRow2Vector(p0, position, triangle[0]);
		util::MatrixRow2Vector(p1, position, triangle[1]);
		util::MatrixRow2Vector(p2, position, triangle[2]);

		Eigen::Vector3d bx = p1 - p0;
		Eigen::Vector3d cx = p2 - p0;
		
		Eigen::Vector3d Ux = bx;
		Ux.normalize();

		Eigen::Vector3d w = Ux.cross(cx);
		Eigen::Vector3d Wx = w;
		Wx.normalize();
		
		Eigen::Vector3d Vx = Ux.cross(Wx);
		Eigen::Matrix3d R = util::GenerateMatrix3DFromColumnVectors(Ux, Vx, Wx);
		Eigen::Vector3d vb = R.transpose() * bx;
		Eigen::Vector3d vc = R.transpose() * cx;

		 material_space << vb[0], vc[0], vb[1], vc[1];

		 double volume = 0.5 * material_space.determinant();
		  if (volume < 0)							 
		 {
			 material_space.row(1) = -1.0 * material_space.row(1);
			 volume = -volume;
		 }
		 assert(volume > 0);
		 volume_[i] = volume;
	} else {
		Eigen::Vector2d point[3];
		util::MatrixRow2Vector(point[0], position, triangle[0]);
		util::MatrixRow2Vector(point[1], position, triangle[1]);
		util::MatrixRow2Vector(point[2], position, triangle[2]);

		material_space = util::GenerateMatrix2DFromColumnVectors(
			point[1] - point[0], point[2] - point[0]);
		volume_[i] = ComputeTriangleVolume(material_space);
	}
	total_volume += volume_[i];


   
	if (mxIsNaN(material_space(0, 0)) || mxIsNaN(material_space(0, 1))
		|| mxIsNaN(material_space(1, 0)) || mxIsNaN(material_space(1, 1))) {
	    #pragma omp critical
			{
				std::cout << "\n MaterialSpace is NaN \n";
				mexErrMsgTxt("NAN value is found");
			}
	}

    inverse_material_space_[i] = material_space.inverse();

    deformation_gradient_differential_[4 * i + 0] =  
        util::GenerateMatrix2DFromRowVectors(kFO, kOO) *
        inverse_material_space_[i];
    deformation_gradient_differential_[4 * i + 1] =
        util::GenerateMatrix2DFromRowVectors(kOO, kFO) *
        inverse_material_space_[i];
    deformation_gradient_differential_[4 * i + 2] =
        util::GenerateMatrix2DFromRowVectors(kSO, kOO) *
        inverse_material_space_[i];
    deformation_gradient_differential_[4 * i + 3] =
        util::GenerateMatrix2DFromRowVectors(kOO, kSO) *
        inverse_material_space_[i];
  }
	size_t vertex_num = position.rows();
	is_stationary_vertex.resize(vertex_num, false);
}



void IsotropicSVDEnergyModel2D::SetDistortionKernel(
    distortion_kernel::DistortionKernel2D* kernel) {
  kernel_ = kernel;
}

double IsotropicSVDEnergyModel2D::ComputeEnergyInBlock(
							const std::vector<Eigen::Vector2d>& position,
							const std::vector<int>& element_block,
						    const data_io::SolverSpecification& solverSpec,
							bool record_invalid_elements) {

	if (solverSpec.is_parallel_energy && solverSpec.non_empty_block_num == 1) {
		return ComputeEnergyInBlockParallel(position, element_block, record_invalid_elements);
	}

	double energy = 0;
	size_t num_of_element  = element_block.size();

	for (auto i : element_block)  {
		Eigen::Vector3i triangle = mesh_[i];

		Eigen::Vector2d point[3] = {
			position[triangle[0]], position[triangle[1]], position[triangle[2]] };

		Eigen::Matrix2d world_space = util::GenerateMatrix2DFromColumnVectors(
			point[1] - point[0], point[2] - point[0]);

		Eigen::Matrix2d deformation_gradient =
			world_space * inverse_material_space_[i];

		if (is_signed_svd)
			util::ComputeSignedSVDForMatrix2D(
				deformation_gradient, &svd_u_[i], &svd_s_[i], &svd_v_[i]);
		else
			util::ComputeSVDForMatrix2D(
				deformation_gradient, &svd_u_[i], &svd_s_[i], &svd_v_[i]);

		
		if (mxIsNaN(svd_s_[i][0]) || mxIsNaN(svd_s_[i][1])) {
			#pragma omp critical 
			{
				std::cout << "\nNAN  Singular values on triangle" << i << std::endl;
				std::cout << "\n  Jacobian" << deformation_gradient << ", \n  World_space =" << world_space;
				mexErrMsgTxt("NAN value is found");
			}
		}

		if (abs(svd_s_[i][0]) < abs(svd_s_[i][1])) {
			#pragma omp critical
			{
				std::cout << "\nAbsolute  Singular values are in wrong order on triangle" << i << std::endl;
			}
		}

		if (record_invalid_elements) {
			is_element_valid[i] = util::IsElementValid(svd_s_[i]);
			invalid_element_num += ! (is_element_valid[i]);
		}

		Eigen::Vector2d kernel_energy = kernel_->ComputeKernelEnergy(svd_s_[i], is_element_valid[i]);

		if (kernel_energy[1] > 0.5) {
			energy = 1e12;
			svd_s_[i] = Eigen::Vector2d(1.0, 1.0);
			svd_u_[i] << 1, 0, 0,1;
			svd_v_[i] << 1, 0, 0, 1;
		}
		energy += volume_[i] * kernel_energy[0];
		element_distortion[i] = kernel_energy[0];
	}

	energy_difference = energy - prev_energy;
	prev_energy = energy;
	return energy;
}


double IsotropicSVDEnergyModel2D::ComputeEnergyInBlockParallel(
	const std::vector<Eigen::Vector2d>& position,
	const std::vector<int>& element_block,
	bool record_invalid_elements) {
	double energy = 0;
	size_t num_of_element = element_block.size();
	bool has_invalid_element = false;
#pragma omp parallel for
	for (int bi = 0; bi < num_of_element; bi++) {
		int i = element_block[bi];
		Eigen::Vector3i triangle = mesh_[i];

		Eigen::Vector2d point[3] = {
			position[triangle[0]], position[triangle[1]], position[triangle[2]] };

		Eigen::Matrix2d world_space = util::GenerateMatrix2DFromColumnVectors(
			point[1] - point[0], point[2] - point[0]);

		Eigen::Matrix2d deformation_gradient =
			world_space * inverse_material_space_[i];

		if (is_signed_svd)
			util::ComputeSignedSVDForMatrix2D(
				deformation_gradient, &svd_u_[i], &svd_s_[i], &svd_v_[i]);
		else
			util::ComputeSVDForMatrix2D(
				deformation_gradient, &svd_u_[i], &svd_s_[i], &svd_v_[i]);

		if (record_invalid_elements) {
			is_element_valid[i] = util::IsElementValid(svd_s_[i]);
		}

		Eigen::Vector2d kernel_energy = kernel_->ComputeKernelEnergy(svd_s_[i], is_element_valid[i]);

		if (kernel_energy[1] > 0.5) {
			has_invalid_element = true;
			//energy = 1e12;
			svd_s_[i] = Eigen::Vector2d(1.0, 1.0);
			svd_u_[i] << 1, 0, 0, 1;
			svd_v_[i] << 1, 0, 0, 1;
		}
		element_distortion[i] = kernel_energy[0];
	}

	if (has_invalid_element) {
		energy = 1e12;
		for (auto i : element_block)
			invalid_element_num += !(is_element_valid[i]);
	} else {
		// parallel reduction
		for (auto i : element_block) {
			invalid_element_num += !(is_element_valid[i]);
			energy += element_distortion[i] * volume_[i];
		}
	}
	energy_difference = energy - prev_energy;
	prev_energy = energy;
	return energy;
}

void IsotropicSVDEnergyModel2D::ComputeGradientInBlock(
	const std::vector<Eigen::Vector2d>& position, Eigen::VectorXd* gradient,
	const std::vector<int>& element_block,
	const std::vector<int>& free_vertex_block,
	data_io::SolverSpecification& solverSpec) {

	size_t num_of_element = element_block.size();

	for (auto vi : free_vertex_block) {
		(*gradient)[2 * vi] = 0;   (*gradient)[2 * vi + 1] = 0;
	}
	
	double dpsi[4];

	for (auto i: element_block) {
		Eigen::Vector3i triangle = mesh_[i];

		Eigen::Vector2d dpsi_ds = kernel_->ComputeKernelGradient(svd_s_[i]);

		for (size_t j = 0; j < 4; j++) {
			Eigen::Matrix2d product = svd_u_[i].transpose() *
				deformation_gradient_differential_[4 * i + j] *
				svd_v_[i];
			ut_df_v[4 * i + j] = product;

			dpsi[j] = volume_[i] *	   
				(product(0, 0) * dpsi_ds[0] + product(1, 1) * dpsi_ds[1]);
		}
		if (mxIsNaN(dpsi[0]) || mxIsNaN(dpsi[1]) || mxIsNaN(dpsi[2]) || mxIsNaN(dpsi[3])) {
			#pragma omp critical 
			{
				std::cout << "\nNAN in dpsi !!! ";
				std::cout << ", sing vals=" << svd_s_[i][0] << ", " << svd_s_[i][1] << std::endl;
				std::cout << "\n  dpsi_ds" << dpsi_ds;
				std::cout << "\n    Jacobian= ";
				for (size_t j = 0; j < 4; j++)
					std::cout << "," << deformation_gradient_differential_[4 * i + j];

				mexErrMsgTxt("NAN value is found");
			}
		}

		(*gradient)[2 * triangle[1] + 0] += dpsi[0];
		(*gradient)[2 * triangle[1] + 1] += dpsi[1];
		(*gradient)[2 * triangle[2] + 0] += dpsi[2];
		(*gradient)[2 * triangle[2] + 1] += dpsi[3];
		(*gradient)[2 * triangle[0] + 0] -= (dpsi[0] + dpsi[2]);
		(*gradient)[2 * triangle[0] + 1] -= (dpsi[1] + dpsi[3]);
	}
}


void IsotropicSVDEnergyModel2D::SetDirichletConstraints(int vertex_num,
														const std::vector<int>& free_vertices,
														const std::vector<int>& fixed_vertices){
	blockFreeVertIndex.resize(vertex_num);
	size_t  free_ver_num = free_vertices.size(),
			fixed_ver_num = fixed_vertices.size();

	for (size_t i = 0; i < free_ver_num; i++) {
		int v = free_vertices[i];
		blockFreeVertIndex[v] = i;
	}
	for (auto v: fixed_vertices) 
		blockFreeVertIndex[v] = -1;
}

void IsotropicSVDEnergyModel2D::ComputeHessianNonzeroEntriesDirConstraintsInBlock(
						const std::vector<Eigen::Vector2d>& position,
						std::vector<Eigen::Triplet<double>>* entry_list,
						const std::vector<int>& element_block,
						data_io::SolverSpecification& solverSpec)
{

	for (auto ele: element_block) {
		Eigen::Vector3i triangle = mesh_[ele];

		Eigen::Matrix<double, 6, 6> element_hessian;
		ComputeElementHessian(position, ele, &element_hessian);
		for (size_t x = 0; x < 3; x++) {
			for (size_t y = 0; y < 3; y++) {
				int m = (x + 2) % 3;
				int n = (y + 2) % 3;
				if (triangle[x] > triangle[y]) {
					continue;
				}
				for (size_t i = 0; i < 2; i++) {
					for (size_t j = 0; j < 2; j++) {
						if (2 * triangle[x] + i > 2 * triangle[y] + j) {
							continue;
						}
						int vert_x = blockFreeVertIndex[triangle[x]],
							vert_y = blockFreeVertIndex[triangle[y]];
						if (vert_x < 0 || vert_y < 0)
							continue;
						entry_list->emplace_back(2 * vert_x + i,
							2 * vert_y + j,
							element_hessian(2 * m + i, 2 * n + j));
					}
				}
			}
		}
	}
	if (entry_list->size() == 0) {
		#pragma omp critical
		{
			std::cout << "Zero or Empty Hessian";
			//mexErrMsgTxt("encounter Zero or Empty Hessian");
		}
	}

}

void IsotropicSVDEnergyModel2D::ComputeElementHessian(
    const std::vector<Eigen::Vector2d>& position,
    int element_index,
    Eigen::Matrix<double, 6, 6>* hessian) {
  Eigen::Matrix2d stretch_eigen_vectors, s0_s1, hessian_eigen_values,
      hessian_eigen_vectors;

  Eigen::Vector2d stretch_eigen_values =
      kernel_->GetStretchPairEigenValues(svd_s_[element_index]);
  kernel_->GetHessianEigenValues(
      svd_s_[element_index], &hessian_eigen_values, &hessian_eigen_vectors);

  Eigen::Matrix<double, 2, 4> hessian_kernel_transform;
  Eigen::Matrix<double, 2, 4> s0_s1_transform;

  for (int i = 0; i < 4; i++) {
    hessian_kernel_transform(0, i) = ut_df_v[4 * element_index + i](0, 0);
    hessian_kernel_transform(1, i) = ut_df_v[4 * element_index + i](1, 1);

    s0_s1_transform(0, i) = ut_df_v[4 * element_index + i](0, 1);
    s0_s1_transform(1, i) = -ut_df_v[4 * element_index + i](1, 0);
  }
  stretch_eigen_vectors << -1.0, 1.0, 1.0, 1.0;
  s0_s1_transform = stretch_eigen_vectors * s0_s1_transform;
  hessian_kernel_transform = hessian_eigen_vectors * hessian_kernel_transform;

  if (enforce_spd_) {
    hessian_eigen_values(0, 0) =
        std::max(hessian_eigen_values(0, 0), spd_projection_threshold_);
    hessian_eigen_values(1, 1) =
        std::max(hessian_eigen_values(1, 1), spd_projection_threshold_);

    stretch_eigen_values[0] =
        std::max(stretch_eigen_values[0], spd_projection_threshold_);
    stretch_eigen_values[1] =
        std::max(stretch_eigen_values[1], spd_projection_threshold_);
  }

  s0_s1 << stretch_eigen_values[0], 0.0, 0.0, stretch_eigen_values[1];

  Eigen::Matrix4d kernel_block =
      hessian_kernel_transform.transpose() * hessian_eigen_values *
          hessian_kernel_transform +
      s0_s1_transform.transpose() * s0_s1 * s0_s1_transform;

  kernel_block = volume_[element_index] * kernel_block;

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      (*hessian)(i, j) = kernel_block(i, j);
    }
  }

  for (size_t i = 0; i < 4; i++) {
    (*hessian)(i, 4) = -((*hessian)(i, 0) + (*hessian)(i, 2));
    (*hessian)(i, 5) = -((*hessian)(i, 1) + (*hessian)(i, 3));
    (*hessian)(4, i) = (*hessian)(i, 4);
    (*hessian)(5, i) = (*hessian)(i, 5);
  }

  for (size_t i = 4; i < 6; i++) {
    for (size_t j = 4; j < 6; j++) {
      (*hessian)(i, j) = -((*hessian)(i - 2, j) + (*hessian)(i - 4, j));
    }
  }
}

}  // namespace mesh_distortion
