// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#include <string>
#include <vector>

#include <mex.h>
#include <Eigen/Dense>

#include "common/util/mesh_obj_io.h"

void mexFunction(int num_of_lhs,
                 mxArray* pointer_of_lhs[],
                 int num_of_rhs,
                 const mxArray* pointer_of_rhs[]) {
  char* file_name;

  file_name = mxArrayToString(pointer_of_rhs[0]);

  mexPrintf("load file %s\n", file_name);

  std::vector<Eigen::Vector3d> vertex;
  std::vector<Eigen::Vector2d> uv;
  std::vector<Eigen::Vector3i> face;

  std::string mesh_file_name(file_name);

  util::ReadObjMeshFile(mesh_file_name, &vertex, &uv, &face);

  mexPrintf("ver #: %d; tri #: %d; uv #: %d.\n",
            vertex.size(),
            face.size(),
            uv.size());

  mxArray *vertex_mex, *face_mex, *uv_mex;

  vertex_mex = pointer_of_lhs[0] =
      mxCreateDoubleMatrix(vertex.size() * 3, 1, mxREAL);
  face_mex = pointer_of_lhs[1] =
      mxCreateDoubleMatrix(face.size() * 3, 1, mxREAL);
  uv_mex = pointer_of_lhs[2] = mxCreateDoubleMatrix(uv.size() * 2, 1, mxREAL);

  double *vertex_output, *face_output, *uv_output;

  vertex_output = mxGetPr(vertex_mex);
  face_output = mxGetPr(face_mex);
  uv_output = mxGetPr(uv_mex);

  for (size_t i = 0; i < vertex.size(); i++) {
    vertex_output[0 * vertex.size() + i] = vertex[i][0];
    vertex_output[1 * vertex.size() + i] = vertex[i][1];
    vertex_output[2 * vertex.size() + i] = vertex[i][2];
  }

  for (size_t i = 0; i < face.size(); i++) {
    face_output[0 * face.size() + i] = face[i][0];
    face_output[1 * face.size() + i] = face[i][1];
    face_output[2 * face.size() + i] = face[i][2];
  }

  for (size_t i = 0; i < uv.size(); i++) {
    uv_output[0 * uv.size() + i] = uv[i][0];
    uv_output[1 * uv.size() + i] = uv[i][1];
  }

  return;
}
