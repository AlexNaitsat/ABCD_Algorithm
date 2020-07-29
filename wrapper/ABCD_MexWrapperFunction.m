%% Copyright @2020. All rights reserved.
%  Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

% Outputs:
% fV       : optimized target coordinates
% meshData : mesh data structure fields for mex solver
function [fV,meshData] = ABCD_MexWrapperFunction(meshData,fixer_spec,optimizer_spec, options)

vn = size(meshData.V,1);
tn = size(meshData.T,1);

% setting mesh dimensions
d = size(meshData.T,2);
if (isfield(options,'is_uv_mesh') && options.is_uv_mesh)
    source_dim = 3;
else
    source_dim = d-1;
end
%% initializng mesh data sructure for mex
meshData.vert_num        =  vn;
meshData.tri_num         =  tn;
meshData.all_elements    =  {0:tn-1}; 
meshData.free_vertices   =  {find(~meshData.is_fixed_vert)'-1};
meshData.fixed_vertices  =  {find(meshData.is_fixed_vert)'-1}; 
meshData.is_fixed_vert   =  double(meshData.is_fixed_vert);    


 optimizer_spec.source_dim = source_dim;
 fixer_spec.source_dim     = source_dim;
 
 optimizer_spec.solver_num = 1;%PN (Do not change solvers)
 fixer_spec.solver_num     = 0;%GD 

if ~isfield(optimizer_spec,'K_hat')
    optimizer_spec.K_hat = 1; %K = infinity
end
optimizer_spec.single_fixed_block=1;

if ~isfield(fixer_spec,'K_hat')
    fixer_spec.K_hat = 1;
end
fixer_spec.single_fixed_block=1;


%% Alternating mex solver
if d==3 %tetrahedral mesh
    fV_column = ABCD_FixerOptimizer_2d_mex(meshData.V(:,1:source_dim), meshData.fV(:,1:2), meshData.T, ...
        vn, tn,fixer_spec,optimizer_spec, ...
        options.iter_num,meshData);
else   %triangle mesh
    fV_column = ABCD_FixerOptimizer_3d_mex(meshData.V(:,1:source_dim), meshData.fV, meshData.T, ...
        vn, tn,fixer_spec,optimizer_spec, ...
        options.iter_num,meshData);
end
fV = reshape(fV_column, d-1, size(fV_column, 1) / (d-1))';
