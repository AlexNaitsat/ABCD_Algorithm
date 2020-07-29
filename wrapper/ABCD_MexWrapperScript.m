%%  Copyright @2020. All rights reserved.
%   Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

clear tmesh
%% loading source/ in mesh
[~,in_name,ext]=fileparts(in_mesh);
tic;
if strcmp(ext,'.obj')
        [V_,T_,UV_]=load_obj_mesh_mex(in_mesh);
        tmesh.T    = reshape(T_, size(T_, 1) / 3,3)+1;
        tmesh.V    = reshape(V_, size(V_, 1) / 3,3);
     if strcmp(target_mesh,'uv')
        tmesh.fV   = reshape(UV_,size(UV_, 1) / 2,2);
     end
elseif  strcmp(ext,'.mat')
    load(in_mesh); %input T,V in mat file
    tmesh.T = T;
    tmesh.V = V;
end
d = size(tmesh.T,2);
if d==3 && isfield(options,'is_uv_mesh') && ~options.is_uv_mesh
     tmesh.source_dim = 2; %remove zero column from planar mesh
else
     tmesh.source_dim = 3;
end
disp(['Loading source mesh ' num2str(toc) ' sec.']);

%% loading target mesh
tic;
[~,target_name,ext]=fileparts(target_mesh);
if strcmp(ext,'.off')
    [fV,~]=read_off_my(target_mesh);
    tmesh.fV=fV';
elseif strcmp(ext,'.obj')
    fV_= load_obj_mesh_mex(target_mesh);
    tmesh.fV    = reshape(fV_, size(fV_, 1) / 3,3);
elseif strcmp(ext,'.mat')
    load(target_mesh);
    tmesh.fV = fV;
    if ~exist('fV','var') && exist('tmesh','var') && isfield(mesh,'fV')
        tmesh.fV = fV;
    end
end

if size(tmesh.fV,2) > d-1
    tmesh.fV = tmesh.fV(:,1:d-1);
end
disp(['Loading initial target coordinates ' num2str(toc) ' sec.']);

%%
tmesh.vn = size(tmesh.V,1);
tmesh.tn = size(tmesh.T,1);
is_1_based_index =0; %0-based index for C++
[tmesh.is_bnd_v, tmesh.vv_mesh_edges, tmesh.vert_neighbors, tmesh.vert_simplices, energy0, elemnent_dist0, sing_val0]  = ...
             GetMeshData_mex(tmesh.T,tmesh.V(:,1:tmesh.source_dim),tmesh.tn, tmesh.vn, 0,tmesh.fV,optimizer_spec);

num_of_flipped   =  sum(prod(sign(sing_val0),2)<0);
num_of_collapsed =  sum(abs(sing_val0(:,end))<10^-8);

%% mesh positional constraints
tmesh.is_fixed_vert = zeros(tmesh.vn,1);
if isfield(options,'fixed_vertices')  
    tmesh.is_fixed_vert(options.fixed_vertices) =1;
end

if ~options.free_bnd %check for fixed boundary
    tmesh.is_fixed_vert   =   tmesh.is_fixed_vert | tmesh.is_bnd_v;  
end
disp(['vertices / anchors # :'   num2str(tmesh.vn) ' / ' num2str(sum(tmesh.is_fixed_vert))  ...
      '; simplices / invalids # :' num2str(tmesh.tn) ' / ' num2str(num_of_flipped+num_of_collapsed) ]);
%% optimization
fV = ABCD_MexWrapperFunction(tmesh,fixer_spec,optimizer_spec,options);
%% visualization
if options.draw_mesh
    if d==3
        figure; trimesh(tmesh.T,tmesh.V(:,1),tmesh.V(:,2),tmesh.V(:,3),'facecolor',[0.7 0.7 0.7], 'edgecolor','black'); view([0 0 1]); axis equal
        title('source');

        log_element_dist0 = log(elemnent_dist0+1);
        [~, ~, ~,~, ~, elemnent_dist, sing_val]  = GetMeshData_mex(tmesh.T,tmesh.V(:,1:tmesh.source_dim), ...
                                                                   tmesh.tn, tmesh.vn, 0,fV,optimizer_spec);
        log_element_dist = log(elemnent_dist+1);

        drawOpt.color_range = [min([log_element_dist; log_element_dist0]) ...
            max([log_element_dist; log_element_dist0]) ];

        figure;
        plot_triangle_mesh(tmesh.T,tmesh.fV, log(elemnent_dist0+0.1), sing_val0, drawOpt, tmesh.is_fixed_vert);
        title('initial target');
            

    figure;
    plot_triangle_mesh(tmesh.T,fV, log(elemnent_dist+0.1), sing_val, drawOpt, tmesh.is_fixed_vert);
    title('final target');
    else
        disp('Drawing tetrahedral meshes ...');
        VV = (1:tmesh.vn)';
        VV(logical(tmesh.is_bnd_v)) =0;
        is_bnd_tet=any(VV(T),2);
        is_fixed=logical(tmesh.is_fixed_vert);
        
        is_flliped  = prod(sign(sing_val0),2)<0;
        figure; tetramesh(T(is_bnd_tet,:),tmesh.fV, ... 
                          log(elemnent_dist0(is_bnd_tet)+0.1),'FaceAlpha',0.3);
        hold on; tetramesh(T(is_flliped,:),tmesh.fV,'FaceAlpha',0.7,'FaceColor','y');
        scatter3(tmesh.fV(is_fixed,1),tmesh.fV(is_fixed,2),tmesh.fV(is_fixed,3), ...
                 100,'blue','filled','MarkerEdgeColor','black');
        title('initial target');
        if isfield(drawOpt,'col_map')
            colormap(drawOpt.col_map);
        else
            colormap cool
        end
        
        figure; tetramesh(tmesh.T(is_bnd_tet,:),fV,'FaceAlpha',0.3,'FaceColor',[.7 .7 .7]);
        hold on;

        scatter3(fV(is_fixed,1),fV(is_fixed,2),fV(is_fixed,3), ...
                 100,'blue','filled','MarkerEdgeColor','black');
        title('final target');
    end
end
%% 
if  ~optimizer_spec.is_parallel  || ~fixer_spec.is_parallel
    disp( ['-W- Slow non-parallel mode!' newline ...
           '    Set optimizer_spac.is_parallel=1 and fixer_spec.is_parallel=1 to optimize  blocks in parallel' newline ...
           '   (also check parallel_energy flag)']);
end
if ~optimizer_spec.use_pardiso
    disp( ['-W- Slow Eigen linear solver!' newline ...
           '    Set optimizer_spec.use_pardiso=1 to use faster Pardiso solver']);
end
