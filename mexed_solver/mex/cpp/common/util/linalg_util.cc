// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)
// anaitsat@campus.technion.ac.il  (Alexander Naitsat)

#include "stdafx.h"
#include <iostream>
#include "common/util/linalg_util.h"

#include <algorithm>

namespace util
{

Eigen::Matrix2d GenerateMatrix2DFromColumnVectors(const Eigen::Vector2d &c0,
												  const Eigen::Vector2d &c1)
{
	Eigen::Matrix2d mat;
	mat.col(0) = c0;
	mat.col(1) = c1;

	return mat;
}

void MatrixRow2Vector(Eigen::Vector2d &Vec, const Eigen::MatrixXd &M, int row)
{
	Vec(0) = M(row, 0);
	Vec(1) = M(row, 1);
}

void MatrixRow2Vector(Eigen::Vector3d &Vec, const Eigen::MatrixXd &M, int row)
{
	Vec(0) = M(row, 0);
	Vec(1) = M(row, 1);
	Vec(2) = M(row, 2);
}

Eigen::Matrix3d GenerateMatrix3DFromColumnVectors(const Eigen::Vector3d &c0,
												  const Eigen::Vector3d &c1,
												  const Eigen::Vector3d &c2)
{
	Eigen::Matrix3d mat;
	mat.col(0) = c0;
	mat.col(1) = c1;
	mat.col(2) = c2;

	return mat;
}

Eigen::Matrix3d GenerateMatrix3DFromRowVectors(const Eigen::Vector3d &r0,
											   const Eigen::Vector3d &r1,
											   const Eigen::Vector3d &r2)
{
	Eigen::Matrix3d mat;
	mat.row(0) = r0;
	mat.row(1) = r1;
	mat.row(2) = r2;

	return mat;
}

Eigen::Matrix2d GenerateMatrix2DFromRowVectors(const Eigen::Vector2d &r0,
											   const Eigen::Vector2d &r1)
{
	Eigen::Matrix2d mat;
	mat.row(0) = r0;
	mat.row(1) = r1;

	return mat;
}

void ComputeSignedSVDForMatrix2D(const Eigen::Matrix2d &mat,
								 Eigen::Matrix2d *u,
								 Eigen::Vector2d *s,
								 Eigen::Matrix2d *v)
{
	Eigen::JacobiSVD<Eigen::Matrix2d> svd(
		mat, Eigen::ComputeFullU | Eigen::ComputeFullV);

	*s = svd.singularValues();
	*u = svd.matrixU();
	*v = svd.matrixV();

	if (u->determinant() < 0.0)
	{
		(*u)(0, 1) *= -1;
		(*u)(1, 1) *= -1;
	}

	if (v->determinant() < 0.0)
	{
		(*v)(0, 1) *= -1;
		(*v)(1, 1) *= -1;
	}

	if (mat.determinant() < 0.0)
	{
		(*s)[1] *= -1;
	}
}

void ComputeSignedSVDForMatrix3D(const Eigen::Matrix3d &mat,
								 Eigen::Matrix3d *u,
								 Eigen::Vector3d *s,
								 Eigen::Matrix3d *v)
{
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(
		mat, Eigen::ComputeFullU | Eigen::ComputeFullV);

	*s = svd.singularValues();
	*u = svd.matrixU();
	*v = svd.matrixV();

	if (u->determinant() < 0.0)
	{
		(*u)(0, 2) *= -1;
		(*u)(1, 2) *= -1;
		(*u)(2, 2) *= -1;
	}

	if (v->determinant() < 0.0)
	{
		(*v)(0, 2) *= -1;
		(*v)(1, 2) *= -1;
		(*v)(2, 2) *= -1;
	}

	if (mat.determinant() < 0.0)
	{
		(*s)[2] *= -1;
	}
}

void ComputeSVDForMatrix2D(const Eigen::Matrix2d &mat,
						   Eigen::Matrix2d *u,
						   Eigen::Vector2d *s,
						   Eigen::Matrix2d *v)
{
	Eigen::JacobiSVD<Eigen::Matrix2d> svd(
		mat, Eigen::ComputeFullU | Eigen::ComputeFullV);

	*s = svd.singularValues();
	*u = svd.matrixU();
	*v = svd.matrixV();

	if (mat.determinant() < 0.0)
	{
		(*s)[1] *= -1;
	}
}


void ComputeSVDForMatrix3D(const Eigen::Matrix3d &mat,
	Eigen::Matrix3d *u,
	Eigen::Vector3d *s,
	Eigen::Matrix3d *v)
{
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(
		mat, Eigen::ComputeFullU | Eigen::ComputeFullV);

	*s = svd.singularValues();
	*u = svd.matrixU();
	*v = svd.matrixV();

	if (mat.determinant() < 0.0)
	{
		(*s)[2] *= -1;
	}
}

} // namespace util
