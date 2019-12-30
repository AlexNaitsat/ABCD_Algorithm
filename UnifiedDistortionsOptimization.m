%% Copyright @2019. All rights reserved.
%  Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

%% main wrapper procedure for  "ABCD_FixerOptimizer_mex" solver that alternates
% between fixer and optimizer stages in ABCD
function fV = UnifiedDistortionsOptimization(mesh,fV0,optParam_list,options,movieOpt)

if ~isfield(options,'draw_colbar')  ;options.draw_colbar=0;   end
if (isfield(options,'is_uv_mesh') && options.is_uv_mesh)
    mesh.source_dim = 3;
else
    mesh.source_dim = 3;
end
if ~isfield(options,'draw_mesh')   ;options.draw_mesh=0;   end
run_mex =1;
options.null = 0;
for optimizer_index=length(optParam_list):-1:1
    if ~strcmp(optParam_list{optimizer_index}.dist_name,'FS')
        break;
    end
end

d = size(mesh.T,2);

if ~isfield(mesh, 'target_pos_normal');mesh.target_pos_normal =[0 0 1]; end %set positive orientation
if d==4
    mesh=add_tet_deformation_3D_analytic(mesh,fV0);
else
    mesh.fV= fV0;
    mesh.SV_t = ones(mesh.tn,2);
end

if options.fV_per_iter
    dataPerIter{1}.fV = fV0;
    dataPerIter{1}.optIndex =0;
end

%%
if ~options.free_bnd
    mesh.is_fixed_v = mesh.is_bnd_v;  %fixing first boundary vertices
else
    mesh.is_fixed_v = zeros(mesh.vn,1);
end
if isfield(options,'fixed_vertices')
    mesh.is_fixed_v(options.fixed_vertices) =1 ; %setting achors
end
%%
T4V  = CellVec(mesh.T4V);
Loc2GlobV= CellVec(mesh.loc2glob_v);

dist_name = options.dist_name;
if isfield(options,'dist_draw') && ~isempty(options.dist_draw)
    dist_draw = options.dist_draw;
else
    dist_draw = dist_name;
end

%Constant fields in mesh data struct
const_mesh.V= mesh.V;
const_mesh.T= mesh.T;
const_mesh.vn=mesh.vn;
const_mesh.tn=mesh.tn;
const_mesh.en=mesh.en;%number of edges
const_mesh.is_bnd_v = mesh.is_bnd_v;
const_mesh.S=  mesh.S;
const_mesh.detT  = mesh.detT;
const_mesh.M=  mesh.M;
const_mesh.WcSum=  mesh.WcSum;
const_mesh.Lc = mesh.Lc;
const_mesh.Wc = mesh.Wc;
const_mesh.SLc = mesh.SLc;
const_mesh.signDf = mesh.signDf;
const_mesh.normT  = mesh.normT;
const_mesh.is_fixed_v = logical(mesh.is_fixed_v);
const_mesh.target_pos_normal =  mesh.target_pos_normal;
const_mesh.perimiter =  mesh.perimiter;

opLS = OptimizationParams.SubStruct(optParam_list{optimizer_index},'LS'); opLS.penalize_flips_t =false(0,1);

if run_mex
    [dist_energy,dE,dist] = dist_gradient_analyticMex_mex(const_mesh,mesh.fV,Loc2GlobV,T4V,opLS);
else
    [dist_energy,dE,dist] = dist_gradient_analyticMex(const_mesh,mesh.fV,Loc2GlobV,T4V,opLS);
end
tot_dist = dist_energy/mesh.WcSum;
mesh.dis_t = dist.t;
mesh.dis_v = dist.v;
dist.grad = dE;
%% Report initial distortions
fliped_num = length(find(dist.SV(:,1)<0));
degen_num  = length(find(abs(dist.SV(:,d-1))<10^-10));
sigularities_str = [num2str(degen_num) '-degenerate simplices'];
sigularities_str =[sigularities_str ')'];

disp('Iteration 0: (matlab report)_____________________________');
disp(['Initial distortion ' dist_name ' = ', num2str(tot_dist), '. ',num2str(fliped_num) '-flipped,' sigularities_str ]);
%%
init_dist = mesh.dis_t;
d_opt = options;
d_opt.dist = 'dis_t';

%%
if movieOpt.single_colmap
    %sorting element distorions
    dist_sort = sort([mesh.dis_t ; init_dist]);
    dlen = length(dist_sort);
    if isfield(options,'colmap_percent')
        percent =options.colmap_percent;
    else
        percent =0.8;
    end
    i0 = 1;
    i1 = floor(dlen*percent) ;
    
    col_min= dist_minimum(options.dist_name,d); %get energy mininumal value
    col_max = dist_sort(i1);
    if (col_min >= col_max)
        if  dist_sort(end) == col_min
            col_max = max(col_max,col_min)+EPS;
        else
            col_max =dist_sort(end);
        end
        
    end
    if isfield(movieOpt,'log_dist_scale') && movieOpt.log_dist_scale
        col_min =log(dist_minimum(options.dist_name,d));
        col_max = log(abs(col_max));
    end
    movieOpt.color_range = [col_min col_max];
    
end

%meshData
meshData.vert_num = const_mesh.vn;
meshData.tri_num = const_mesh.tn;
meshData.vert_simplices =  cellfun(@(X) X-1,mesh.T4V,'UniformOutput',false);
meshData.vert_neighbors =  cellfun(@(X) X-1,mesh.V4V,'UniformOutput',false);
meshData.is_fixed_vert    =  double(const_mesh.is_fixed_v);
meshData.vv_mesh_edges = mesh.Eu-1;
meshData.all_elements  =  {0:mesh.tn-1}; %var_mesh.all_elements;
meshData.free_vertices =   {find(~mesh.is_fixed_v)'-1};%var_mesh.free_vertices;
meshData.fixed_vertices = {find(mesh.is_fixed_v)'-1}; %var_mesh.fixed_vertices;

if (isfield(options,'is_uv_mesh') && options.is_uv_mesh)
    var_mesh.source_dim = 3;
else
    var_mesh.source_dim = size(mesh.T,2)-1;
end

movieOpt.Loc2GlobV = Loc2GlobV;
movieOpt.T4V = T4V;
%% Visualizing initial mapping
title_str = ['Initialization (' dist_name ')'];

if options.draw_mesh
    if ~strcmp(dist_draw,'None')
        clf;
    end
    figure;
    colormap jet
    hold off
    if d==4
        draw_opt_results_3D_movie_analytic(mesh,mesh.fV,dist_draw,dist, movieOpt);
    else
        draw_opt_results_Surf_analytic(mesh,mesh.fV, dist_draw,dist, movieOpt);
    end
    if isfield(movieOpt,'draw_rec') && length(movieOpt.draw_rec) > 1
        hold on;
        draw_bbox_3D(movieOpt.draw_rec(1,:),movieOpt.draw_rec(2,:),movieOpt.rec_LineSpec);
    end
    if movieOpt.single_colmap
        caxis manual; caxis([col_min col_max]);
    end
    if options.draw_colbar; colorbar; end
    
    title(title_str,'FontSize',16,'interpreter','latex');
    hold on
end
mesh.fixed_elements = find(all(mesh.is_fixed_v(mesh.T),2));  %matlab index

%% Runnning all ABCD  iterations inside the alternating mex solver
optimizer_spec = GetMexedSolverSpec(optParam_list{2});
fixer_spec = GetMexedSolverSpec(optParam_list{1});
optimizer_spec.source_dim = mesh.source_dim;
fixer_spec.source_dim     = mesh.source_dim;

tic
optimizer_spec.K_hat = 1;
optimizer_spec.single_fixed_block=0;
fixer_spec.K_hat = 1;
fixer_spec.single_fixed_block=0;

fixer_spec.is_parallel_energy =1;   optimizer_spec.is_parallel_energy =1;
fixer_spec.is_parallel_grad =0;     optimizer_spec.is_parallel_grad =0;
fixer_spec.is_parallel_hessian =0;  optimizer_spec.is_parallel_hessian =0;



fV_column = ABCD_FixerOptimizer_mex(mesh.V(:,1:mesh.source_dim), mesh.fV(:,1:2), mesh.T, ...
    mesh.vn, mesh.tn,fixer_spec,optimizer_spec, ...
    options.iter_num,meshData);
fV = reshape(fV_column, d-1, size(fV_column, 1) / (d-1))';
%% visualizing results
if options.draw_mesh
    figure;
    opLS=OptimizationParams.SubStruct(optParam_list{optimizer_index},'LS');
    opLS.penalize_flips_t = false(0,1);
    is_fixed_v = const_mesh.is_fixed_v;
    const_mesh.is_fixed_v(:)=0;
    if run_mex
        [dist_energy,grad,distS]=dist_gradient_analyticMex_mex(const_mesh,fV,Loc2GlobV,T4V,opLS );
    else
        [dist_energy,grad,distS]=    dist_gradient_analyticMex(const_mesh,fV,Loc2GlobV,T4V,opLS );
    end
    tot_dist = dist_energy/mesh.WcSum;
    %const_mesh.is_fixed_v = is_fixed_v;
    
    if d==4
        draw_opt_results_3D_movie_analytic(mesh,fV,dist_draw,distS, movieOpt);
    else
        draw_opt_results_Surf_analytic(mesh,fV,dist_draw,distS, movieOpt);
    end
    if isfield(movieOpt,'draw_rec') && length(movieOpt.draw_rec) > 1
        hold on;
        draw_bbox_3D(movieOpt.draw_rec(1,:),movieOpt.draw_rec(2,:),movieOpt.rec_LineSpec);
    end
    
    if movieOpt.single_colmap
        caxis manual; caxis([col_min col_max]);
    end
    if options.draw_colbar; colorbar; end
    out_iter= options.iter_num;
    title_str=['Img$(f_{' num2str(out_iter) '})$ (' dist_name ')'];
    title(title_str,'FontSize',16,'interpreter','latex');
    
    fliped_num = length(find(distS.SV(:,1)<0));
    degen_num  = length(find(abs(distS.SV(:,d-1))<10^-10));
    sigularities_str = [num2str(degen_num) '-degenerate simplices'];
    sigularities_str =[sigularities_str ')'];
    
    disp('Final results (matlab report)____________________________________');
    disp(['Total distortion ' dist_name '= ', num2str(tot_dist), '. ',num2str(fliped_num) '-flipped,' sigularities_str ]);
end