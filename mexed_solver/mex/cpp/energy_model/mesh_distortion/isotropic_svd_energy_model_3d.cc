// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#include "stdafx.h"
#include "energy_model/mesh_distortion/isotropic_svd_energy_model_3d.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>

#include "common/util/linalg_util.h"
#include "data_io/data_io_utils.h"
namespace mesh_distortion
{

namespace
{

double ComputeTetrahedraVolume(const Eigen::Matrix3d &mat)
{
  double volume = mat.determinant() / 6.0;
  assert(volume > 0.0);
  return volume;
}

} // namespace

void IsotropicSVDEnergyModel3D::SetRestMesh(
    const Eigen::MatrixXd &position,
    const std::vector<Eigen::Vector4i> &mesh)
{
  size_t num_of_element = mesh.size();
  mesh_ = mesh;
  volume_.resize(num_of_element);
  inverse_material_space_.resize(num_of_element);
  deformation_gradient_differential_.resize(9 * num_of_element);
  ut_df_v.resize(9 * num_of_element);
  svd_u_.resize(num_of_element);
  svd_v_.resize(num_of_element);
  svd_s_.resize(num_of_element);

  element_distortion.resize(num_of_element);
  element_distortion.setZero();

  const Eigen::Vector3d kOO(0.0, 0.0, 0.0);
  const Eigen::Vector3d kFO(1.0, 0.0, 0.0);
  const Eigen::Vector3d kSO(0.0, 1.0, 0.0);
  const Eigen::Vector3d kTO(0.0, 0.0, 1.0);

  for (size_t i = 0; i < num_of_element; i++)
  {
    Eigen::Vector4i tetrahedron = mesh[i];

    Eigen::Vector3d point[4];

    util::MatrixRow2Vector(point[0], position, tetrahedron[0]);
    util::MatrixRow2Vector(point[1], position, tetrahedron[1]);
    util::MatrixRow2Vector(point[2], position, tetrahedron[2]);
    util::MatrixRow2Vector(point[3], position, tetrahedron[3]);

    Eigen::Matrix3d material_space = util::GenerateMatrix3DFromColumnVectors(
        point[1] - point[0], point[2] - point[0], point[3] - point[0]);

    volume_[i] = ComputeTetrahedraVolume(material_space);
    inverse_material_space_[i] = material_space.inverse();

    deformation_gradient_differential_[9 * i + 0] =
        util::GenerateMatrix3DFromRowVectors(kFO, kOO, kOO) *
        inverse_material_space_[i];
    deformation_gradient_differential_[9 * i + 1] =
        util::GenerateMatrix3DFromRowVectors(kOO, kFO, kOO) *
        inverse_material_space_[i];
    deformation_gradient_differential_[9 * i + 2] =
        util::GenerateMatrix3DFromRowVectors(kOO, kOO, kFO) *
        inverse_material_space_[i];
    deformation_gradient_differential_[9 * i + 3] =
        util::GenerateMatrix3DFromRowVectors(kSO, kOO, kOO) *
        inverse_material_space_[i];
    deformation_gradient_differential_[9 * i + 4] =
        util::GenerateMatrix3DFromRowVectors(kOO, kSO, kOO) *
        inverse_material_space_[i];
    deformation_gradient_differential_[9 * i + 5] =
        util::GenerateMatrix3DFromRowVectors(kOO, kOO, kSO) *
        inverse_material_space_[i];
    deformation_gradient_differential_[9 * i + 6] =
        util::GenerateMatrix3DFromRowVectors(kTO, kOO, kOO) *
        inverse_material_space_[i];
    deformation_gradient_differential_[9 * i + 7] =
        util::GenerateMatrix3DFromRowVectors(kOO, kTO, kOO) *
        inverse_material_space_[i];
    deformation_gradient_differential_[9 * i + 8] =
        util::GenerateMatrix3DFromRowVectors(kOO, kOO, kTO) *
        inverse_material_space_[i];
  }
}

void IsotropicSVDEnergyModel3D::SetDistortionKernel(
    distortion_kernel::DistortionKernel3D *kernel)
{
  kernel_ = kernel;
}

double IsotropicSVDEnergyModel3D::ComputeEnergy(
    const std::vector<Eigen::Vector3d> &position)
{
  double energy = 0;
  size_t num_of_element = mesh_.size();

  for (size_t i = 0; i < num_of_element; i++)
  {
    Eigen::Vector4i tetrahedron = mesh_[i];

    Eigen::Vector3d point[4] = {position[tetrahedron[0]],
                                position[tetrahedron[1]],
                                position[tetrahedron[2]],
                                position[tetrahedron[3]]};

    Eigen::Matrix3d world_space = util::GenerateMatrix3DFromColumnVectors(
        point[1] - point[0], point[2] - point[0], point[3] - point[0]);

    Eigen::Matrix3d deformation_gradient =
        world_space * inverse_material_space_[i];

    util::ComputeSignedSVDForMatrix3D(
        deformation_gradient, &svd_u_[i], &svd_s_[i], &svd_v_[i]);

    Eigen::Vector2d kernel_energy = kernel_->ComputeKernelEnergy(svd_s_[i]);

    if (kernel_energy[1] > 0.5)
    {
      energy = 1e12;
      break;
    }
    energy += volume_[i] * kernel_energy[0];
  }

  return energy;
}

double IsotropicSVDEnergyModel3D::ComputeEnergyInBlock(
    const std::vector<Eigen::Vector3d> &position,
    const std::vector<int> &element_block)
{
  double energy = 0;
  size_t num_of_element = element_block.size();

  for (auto i : element_block)
  {
    Eigen::Vector4i tetrahedron = mesh_[i];

    Eigen::Vector3d point[4] = {position[tetrahedron[0]],
                                position[tetrahedron[1]],
                                position[tetrahedron[2]],
                                position[tetrahedron[3]]};

    Eigen::Matrix3d world_space = util::GenerateMatrix3DFromColumnVectors(
        point[1] - point[0], point[2] - point[0], point[3] - point[0]);

    Eigen::Matrix3d deformation_gradient =
        world_space * inverse_material_space_[i];

    if (is_signed_svd)
      util::ComputeSignedSVDForMatrix3D(
          deformation_gradient, &svd_u_[i], &svd_s_[i], &svd_v_[i]);
    else
      util::ComputeSVDForMatrix3D(
          deformation_gradient, &svd_u_[i], &svd_s_[i], &svd_v_[i]);

    Eigen::Vector2d kernel_energy = kernel_->ComputeKernelEnergy(svd_s_[i]);

    if (kernel_energy[1] > 0.5)
    {
      energy = 1e12;
      break;
    }
    energy += volume_[i] * kernel_energy[0];
    element_distortion[i] = kernel_energy[0];
  }

  return energy;
}

void IsotropicSVDEnergyModel3D::ComputeGradient(
    const std::vector<Eigen::Vector3d> &position, Eigen::VectorXd *gradient)
{
  size_t num_of_element = mesh_.size();

  gradient->resize(3 * position.size());
  gradient->setZero();

  double dpsi[9];

  for (size_t i = 0; i < num_of_element; i++)
  {
    Eigen::Vector4i tetrahedron = mesh_[i];

    Eigen::Vector3d dpsi_ds = kernel_->ComputeKernelGradient(svd_s_[i]);

    for (size_t j = 0; j < 9; j++)
    {
      Eigen::Matrix3d product = svd_u_[i].transpose() *
                                deformation_gradient_differential_[9 * i + j] *
                                svd_v_[i];
      ut_df_v[9 * i + j] = product;

      dpsi[j] = volume_[i] *
                (product(0, 0) * dpsi_ds[0] + product(1, 1) * dpsi_ds[1] +
                 product(2, 2) * dpsi_ds[2]);
    }

    (*gradient)[3 * tetrahedron[1] + 0] += dpsi[0];
    (*gradient)[3 * tetrahedron[1] + 1] += dpsi[1];
    (*gradient)[3 * tetrahedron[1] + 2] += dpsi[2];

    (*gradient)[3 * tetrahedron[2] + 0] += dpsi[3];
    (*gradient)[3 * tetrahedron[2] + 1] += dpsi[4];
    (*gradient)[3 * tetrahedron[2] + 2] += dpsi[5];

    (*gradient)[3 * tetrahedron[3] + 0] += dpsi[6];
    (*gradient)[3 * tetrahedron[3] + 1] += dpsi[7];
    (*gradient)[3 * tetrahedron[3] + 2] += dpsi[8];

    (*gradient)[3 * tetrahedron[0] + 0] -= (dpsi[0] + dpsi[3] + dpsi[6]);
    (*gradient)[3 * tetrahedron[0] + 1] -= (dpsi[1] + dpsi[4] + dpsi[7]);
    (*gradient)[3 * tetrahedron[0] + 2] -= (dpsi[2] + dpsi[5] + dpsi[8]);
  }
}

void IsotropicSVDEnergyModel3D::ComputeGradientInBlock(
    const std::vector<Eigen::Vector3d> &position, Eigen::VectorXd *gradient,
    const std::vector<int> &element_block,
    const std::vector<int> &free_vertex_block)
{
  size_t num_of_element = element_block.size();

  for (auto vi : free_vertex_block)
  {
    (*gradient)[3 * vi] = 0;
    (*gradient)[3 * vi + 1] = 0;
    (*gradient)[3 * vi + 2] = 0;
  }
  double dpsi[9];

  for (auto i : element_block)
  {
    Eigen::Vector4i tetrahedron = mesh_[i];

    Eigen::Vector3d dpsi_ds = kernel_->ComputeKernelGradient(svd_s_[i]);

    for (size_t j = 0; j < 9; j++)
    {
      Eigen::Matrix3d product = svd_u_[i].transpose() *
                                deformation_gradient_differential_[9 * i + j] *
                                svd_v_[i];
      ut_df_v[9 * i + j] = product;

      dpsi[j] = volume_[i] *
                (product(0, 0) * dpsi_ds[0] + product(1, 1) * dpsi_ds[1] +
                 product(2, 2) * dpsi_ds[2]);
    }

    (*gradient)[3 * tetrahedron[1] + 0] += dpsi[0];
    (*gradient)[3 * tetrahedron[1] + 1] += dpsi[1];
    (*gradient)[3 * tetrahedron[1] + 2] += dpsi[2];

    (*gradient)[3 * tetrahedron[2] + 0] += dpsi[3];
    (*gradient)[3 * tetrahedron[2] + 1] += dpsi[4];
    (*gradient)[3 * tetrahedron[2] + 2] += dpsi[5];

    (*gradient)[3 * tetrahedron[3] + 0] += dpsi[6];
    (*gradient)[3 * tetrahedron[3] + 1] += dpsi[7];
    (*gradient)[3 * tetrahedron[3] + 2] += dpsi[8];

    (*gradient)[3 * tetrahedron[0] + 0] -= (dpsi[0] + dpsi[3] + dpsi[6]);
    (*gradient)[3 * tetrahedron[0] + 1] -= (dpsi[1] + dpsi[4] + dpsi[7]);
    (*gradient)[3 * tetrahedron[0] + 2] -= (dpsi[2] + dpsi[5] + dpsi[8]);
  }
}

void IsotropicSVDEnergyModel3D::ComputeHessian(
    const std::vector<Eigen::Vector3d> &position,
    Eigen::SparseMatrix<double> *hessian)
{
  size_t num_of_vertex = position.size();

  (*hessian) =
      Eigen::SparseMatrix<double>(3 * num_of_vertex, 3 * num_of_vertex);

  std::vector<Eigen::Triplet<double>> entry_list;

  ComputeHessianNonzeroEntries(position, &entry_list);

  hessian->setFromTriplets(entry_list.begin(), entry_list.end());
}

void IsotropicSVDEnergyModel3D::ComputeHessianNonzeroEntries(
    const std::vector<Eigen::Vector3d> &position,
    std::vector<Eigen::Triplet<double>> *entry_list)
{
  size_t num_of_element = mesh_.size();

  for (size_t ele = 0; ele < num_of_element; ele++)
  {
    Eigen::Vector4i tetrahedron = mesh_[ele];

    Eigen::Matrix<double, 12, 12> element_hessian;
    ComputeElementHessian(position, ele, &element_hessian);

    for (size_t x = 0; x < 4; x++)
    {
      for (size_t y = 0; y < 4; y++)
      {
        if (tetrahedron[x] < tetrahedron[y])
        {
          continue;
        }
        int m = (x + 3) % 4;
        int n = (y + 3) % 4;

        for (size_t i = 0; i < 3; i++)
        {
          for (size_t j = 0; j < 3; j++)
          {
            if (3 * tetrahedron[x] + i < 3 * tetrahedron[y] + j)
            {
              continue;
            }
            entry_list->emplace_back(3 * tetrahedron[x] + i,
                                     3 * tetrahedron[y] + j,
                                     element_hessian(3 * m + i, 3 * n + j));
          }
        }
      }
    }
  }
}

void IsotropicSVDEnergyModel3D::ComputeElementHessian(
    const std::vector<Eigen::Vector3d> &position,
    int element_index,
    Eigen::Matrix<double, 12, 12> *hessian)
{
  Eigen::Matrix3d hessian_eigen_values, hessian_eigen_vectors;
  Eigen::Matrix2d stretch_eigen_vectors, s0_s1, s1_s2, s2_s0;

  Eigen::VectorXd stretch_eigen_values =
      kernel_->GetStretchPairEigenValues(svd_s_[element_index]);
  kernel_->GetHessianEigenValues(
      svd_s_[element_index], &hessian_eigen_values, &hessian_eigen_vectors);

  Eigen::Matrix<double, 3, 9> hessian_kernel_transform;
  Eigen::Matrix<double, 2, 9> s0_s1_transform;
  Eigen::Matrix<double, 2, 9> s1_s2_transform;
  Eigen::Matrix<double, 2, 9> s2_s0_transform;

  for (int i = 0; i < 9; i++)
  {
    hessian_kernel_transform(0, i) = ut_df_v[9 * element_index + i](0, 0);
    hessian_kernel_transform(1, i) = ut_df_v[9 * element_index + i](1, 1);
    hessian_kernel_transform(2, i) = ut_df_v[9 * element_index + i](2, 2);

    s0_s1_transform(0, i) = ut_df_v[9 * element_index + i](0, 1);
    s0_s1_transform(1, i) = -ut_df_v[9 * element_index + i](1, 0);

    // Could be wrong
    s1_s2_transform(0, i) = ut_df_v[9 * element_index + i](1, 2);
    s1_s2_transform(1, i) = -ut_df_v[9 * element_index + i](2, 1);

    s2_s0_transform(0, i) = ut_df_v[9 * element_index + i](2, 0);
    s2_s0_transform(1, i) = -ut_df_v[9 * element_index + i](0, 2);
  }
  stretch_eigen_vectors << -1.0, 1.0, 1.0, 1.0;
  s0_s1_transform = stretch_eigen_vectors * s0_s1_transform;
  s1_s2_transform = stretch_eigen_vectors * s1_s2_transform;
  s2_s0_transform = stretch_eigen_vectors * s2_s0_transform;
  hessian_kernel_transform = hessian_eigen_vectors * hessian_kernel_transform;

  if (enforce_spd_)
  {
    hessian_eigen_values(0, 0) =
        std::max(hessian_eigen_values(0, 0), spd_projection_threshold_);
    hessian_eigen_values(1, 1) =
        std::max(hessian_eigen_values(1, 1), spd_projection_threshold_);
    hessian_eigen_values(2, 2) =
        std::max(hessian_eigen_values(2, 2), spd_projection_threshold_);

    stretch_eigen_values[0] =
        std::max(stretch_eigen_values[0], spd_projection_threshold_);
    stretch_eigen_values[1] =
        std::max(stretch_eigen_values[1], spd_projection_threshold_);

    stretch_eigen_values[2] =
        std::max(stretch_eigen_values[2], spd_projection_threshold_);
    stretch_eigen_values[3] =
        std::max(stretch_eigen_values[3], spd_projection_threshold_);

    stretch_eigen_values[4] =
        std::max(stretch_eigen_values[4], spd_projection_threshold_);
    stretch_eigen_values[5] =
        std::max(stretch_eigen_values[5], spd_projection_threshold_);
  }

  s0_s1 << stretch_eigen_values[0], 0.0, 0.0, stretch_eigen_values[1];
  s1_s2 << stretch_eigen_values[2], 0.0, 0.0, stretch_eigen_values[3];
  s2_s0 << stretch_eigen_values[4], 0.0, 0.0, stretch_eigen_values[5];

  Eigen::Matrix<double, 9, 9> kernel_block =
      hessian_kernel_transform.transpose() * hessian_eigen_values *
          hessian_kernel_transform +
      s0_s1_transform.transpose() * s0_s1 * s0_s1_transform +
      s1_s2_transform.transpose() * s1_s2 * s1_s2_transform +
      s2_s0_transform.transpose() * s2_s0 * s2_s0_transform;

  kernel_block = volume_[element_index] * kernel_block;

  for (size_t i = 0; i < 9; i++)
  {
    for (size_t j = 0; j < 9; j++)
    {
      (*hessian)(i, j) = kernel_block(i, j);
    }
  }

  for (size_t i = 0; i < 9; i++)
  {
    (*hessian)(i, 9) =
        -((*hessian)(i, 0) + (*hessian)(i, 3) + (*hessian)(i, 6));
    (*hessian)(i, 10) =
        -((*hessian)(i, 1) + (*hessian)(i, 4) + (*hessian)(i, 7));
    (*hessian)(i, 11) =
        -((*hessian)(i, 2) + (*hessian)(i, 5) + (*hessian)(i, 8));
    (*hessian)(9, i) = (*hessian)(i, 9);
    (*hessian)(10, i) = (*hessian)(i, 10);
    (*hessian)(11, i) = (*hessian)(i, 11);
  }

  for (size_t i = 9; i < 12; i++)
  {
    for (size_t j = 9; j < 12; j++)
    {
      (*hessian)(i, j) =
          -((*hessian)(i - 3, j) + (*hessian)(i - 6, j) + (*hessian)(i - 9, j));
    }
  }
}

void IsotropicSVDEnergyModel3D::ComputeHessianNonzeroEntriesDirConstraintsInBlock(
    const std::vector<Eigen::Vector3d> &position,
    std::vector<Eigen::Triplet<double>> *entry_list,
    const std::vector<int> &element_block)
{

  for (auto ele : element_block)
  {
    Eigen::Vector4i tetrahedron = mesh_[ele];

    Eigen::Matrix<double, 12, 12> element_hessian;
    ComputeElementHessian(position, ele, &element_hessian);

    for (size_t x = 0; x < 4; x++)
    {
      for (size_t y = 0; y < 4; y++)
      {
        int m = (x + 3) % 4;
        int n = (y + 3) % 4;
        if (tetrahedron[x] < tetrahedron[y])
        {
          continue;
        }
        for (size_t i = 0; i < 3; i++)
        {
          for (size_t j = 0; j < 3; j++)
          {
            if (3 * tetrahedron[x] + i < 3 * tetrahedron[y] + j)
            {
              continue;
            }
            int vert_x = blockFreeVertIndex[tetrahedron[x]],
                vert_y = blockFreeVertIndex[tetrahedron[y]];
            if (vert_x < 0 || vert_y < 0)
              continue;
            entry_list->emplace_back(3 * vert_x + i,
                                     3 * vert_y + j,
                                     element_hessian(3 * m + i, 3 * n + j));
          }
        }
      }
    }
  }
  if (entry_list->size() == 0)
  {
    std::cout << "Zero or Empty Hessian";
    mexErrMsgTxt("encounter Zero or Empty Hessian");
  }
}

} // namespace mesh_distortion
