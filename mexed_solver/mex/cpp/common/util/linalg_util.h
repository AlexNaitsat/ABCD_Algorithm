// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#pragma once

#include "Eigen/Dense"

namespace util
{

Eigen::Matrix2d GenerateMatrix2DFromColumnVectors(const Eigen::Vector2d& c0,
                                                  const Eigen::Vector2d& c1);

Eigen::Matrix2d GenerateMatrix2DFromRowVectors(const Eigen::Vector2d& r0,
                                               const Eigen::Vector2d& r1);

Eigen::Matrix3d GenerateMatrix3DFromColumnVectors(const Eigen::Vector3d& c0,
												  const Eigen::Vector3d& c1,
												  const Eigen::Vector3d& c2);

void  MatrixRow2Vector(Eigen::Vector2d& Vec, const Eigen::MatrixXd& M, int row);
void  MatrixRow2Vector(Eigen::Vector3d& Vec, const Eigen::MatrixXd& M, int row);


void ComputeSignedSVDForMatrix2D(const Eigen::Matrix2d& mat,
                                 Eigen::Matrix2d* u,
                                 Eigen::Vector2d* s,
                                 Eigen::Matrix2d* v);


Eigen::Matrix3d GenerateMatrix3DFromRowVectors(const Eigen::Vector3d& r0,
                                               const Eigen::Vector3d& r1,
                                               const Eigen::Vector3d& r2);
											   
void ComputeSVDForMatrix2D(const Eigen::Matrix2d& mat,
	Eigen::Matrix2d* u,
	Eigen::Vector2d* s,
	Eigen::Matrix2d* v);

inline bool IsElementValid(const Eigen::Vector2d& s) {
	return (s[1] >  1e-12 && s[0] >  1e-12);
}

inline bool IsElementValid(const Eigen::Vector3d& s) {
	return (s[2] >  1e-12 && s[1] >  1e-12 && s[0] >  1e-12);
}

	
void ComputeSignedSVDForMatrix3D(const Eigen::Matrix3d &mat,
                                 Eigen::Matrix3d *u,
                                 Eigen::Vector3d *s,
                                 Eigen::Matrix3d *v);

void ComputeSVDForMatrix3D(const Eigen::Matrix3d &mat,
								 Eigen::Matrix3d *u,
								 Eigen::Vector3d *s,
								 Eigen::Matrix3d *v);

}  // namespace util
