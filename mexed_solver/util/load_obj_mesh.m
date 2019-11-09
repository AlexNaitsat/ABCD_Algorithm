function [vertex, face, uv] = load_obj_mesh( file_name )

[vertex, face, uv] = load_obj_mesh_mex(strcat(file_name, '.obj'));

face = face + 1;

vertex_num = size(vertex, 1) / 3;
triangle_num = size(face, 1) / 3;

vertex = reshape(vertex, vertex_num, 3);
face = reshape(face, triangle_num, 3);

end

