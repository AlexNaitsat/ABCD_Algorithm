// Copyright @2019. All rights reserved.
// Authors: mike323zyf@gmail.com (Yufeng Zhu)

#include "common/util/mesh_obj_io.h"

#include "igl/readOBJ.h"

namespace util {

void ReadObjMeshFile(std::string file_name,
                     std::vector<Eigen::Vector3d>* position,
                     std::vector<Eigen::Vector2d>* uv,
                     std::vector<Eigen::Vector3i>* mesh) {
  Eigen::MatrixXd vertex, texture_coordinate, normal;
  Eigen::MatrixXi face, face_texture_coordinate, face_normal;

  igl::readOBJ(file_name,
               vertex,
               texture_coordinate,
               normal,
               face,
               face_texture_coordinate,
               face_normal);

  size_t num_of_vertices = vertex.rows();
  size_t num_of_elements = face.rows();
  size_t num_of_uvs = texture_coordinate.rows();

  position->resize(num_of_vertices);
  uv->resize(num_of_uvs);
  mesh->resize(num_of_elements);

  for (size_t i = 0; i < num_of_vertices; i++) {
    (*position)[i] = vertex.row(i);
  }

  for (size_t i = 0; i < num_of_uvs; i++) {
    (*uv)[i] = texture_coordinate.row(i);
  }

  for (size_t i = 0; i < num_of_elements; i++) {
    (*mesh)[i] = face.row(i);
  }
}

void WriteObjMeshFile(std::string file_name,
                      const std::vector<Eigen::Vector3d>& position,
                      const std::vector<Eigen::Vector2d>& uv,
                      const std::vector<Eigen::Vector3i>& mesh) {}

}  // namespace util
