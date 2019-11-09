function [vertex_optimized,grad,search_dir,SV_t,dist_t,E,runtimes] = ABCD_mex_solver( ...
                                                position_rest, position_deformed,...
                                                face, optimization_spec,fixed_elements,...
                                                meshData)

num_of_vertex = size(position_rest, 1);
num_of_element = size(face, 1);
optimization_spec.return_search_dir=1;

%% NEW interface without block data inputs
runtimes  =zeros(7,1);

  
[vertex_optimized,grad,search_dir,SV_t,dist_t,E,runtimes] = ABCD_optimize_single_measure_mex(...
                                                            position_rest, position_deformed, face, ...
                                                         num_of_vertex, num_of_element,optimization_spec,meshData);

%%
search_dir = reshape(search_dir, 2, size(search_dir, 1) / 2)';
grad       = reshape(grad, 2, size(grad, 1) / 2)';
vertex_optimized = reshape(vertex_optimized, 2, size(vertex_optimized, 1) / 2)';
dist_t(fixed_elements) =0; 
end
