%% Copyright @2019. All rights reserved.
%  Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

%% main procedure that alternates between optimizer and fixer stages
function fV = UnifiedDistortionsOptimizationIterVisualize(mesh,fV0,optParam_list,options,movieOpt)

run_mex =1; 
options.null = 0;
mov_i = 1;
for optimizer_index=length(optParam_list):-1:1
    if ~strcmp(optParam_list{optimizer_index}.dist_name,'FS')
        break;
    end
end
dis_field = 'dis_v'; 
if ~options.show_v_dist
    disp_dist = [dis_field(1:end-1) 't'];
else
    disp_dist = dis_field;
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
V4V  = CellVec(mesh.V4V);
opF4V= CellMat(mesh.opF4V,1);
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
    
disp('iteration 0 __________________________________________');
disp(['Initial total ' dist_name ' dist= ', num2str(tot_dist), '. ',num2str(fliped_num) '-flipped,' sigularities_str ]);
%%
init_dist = mesh.(dis_field);
d_opt = options;
d_opt.dist = [dis_field(1:end-1) 't']; 

%% 
if options.draw_mesh
    fig=figure;%
    fig.Color = movieOpt.bg_color;
    if ~movieOpt.show_axis
        axis off
    end
    if movieOpt.maximize_fig
        set(gcf, 'Position', get(0,'Screensize'));
    end
end
%%
if movieOpt.single_colmap
    %sorting element distorions 
    dist_sort = sort([mesh.(disp_dist) ; init_dist]); 
    dlen = length(dist_sort);
    if isfield(options,'colmap_percent')
        percent =options.colmap_percent;
    else
        percent =0.8;
    end
    i0 = 1; 
    i1 = floor(dlen*percent) ;
    
    col_min= dist_minimum(options.dist_name,d); %get energy mininumal value
    if options.show_v_dist; col_min = dist_sort(i0); end
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
%Varying fields in mesh  data struct
var_mesh.fV    = mesh.fV;     
var_mesh.rad_v = mesh.rad_v;  
var_mesh.dis_v = mesh.dis_v;
var_mesh.dis_t = mesh.dis_t;
var_mesh.SV_t  = mesh.SV_t;
var_mesh.kernelBasis =  mesh.kernelBasis;

% mexed solver fields
var_mesh.V4V =mesh.V4V;  
var_mesh.T4V =mesh.T4V;
var_mesh.fixed_elements = find(all(mesh.is_fixed_v(mesh.T),2));  %matlab index
var_mesh.is_fixed_element = all(mesh.is_fixed_v(mesh.T),2); 
var_mesh.Eu  =mesh.Eu; %unique edges, used in block partitioning mex 
var_mesh.free_vertices  =  {find(~mesh.is_fixed_v)'-1}; %mex index
var_mesh.fixed_vertices  = {find(mesh.is_fixed_v)'-1};
var_mesh.all_elements   = {0:mesh.tn-1};
%meshData
meshData.vert_num = const_mesh.vn;
meshData.tri_num = const_mesh.tn;
meshData.vert_simplices =  cellfun(@(X) X-1,mesh.T4V,'UniformOutput',false);
meshData.vert_neighbors =  cellfun(@(X) X-1,mesh.V4V,'UniformOutput',false);
meshData.is_fixed_vert    =  double(const_mesh.is_fixed_v);
meshData.vv_mesh_edges = mesh.Eu-1;
meshData.all_elements  = var_mesh.all_elements;
meshData.free_vertices = var_mesh.free_vertices;
meshData.fixed_vertices = var_mesh.fixed_vertices;

if (isfield(options,'is_uv_mesh') && options.is_uv_mesh)
    var_mesh.source_dim = 3;
else
    var_mesh.source_dim = size(mesh.T,2)-1;
end
        
movieOpt.Loc2GlobV = Loc2GlobV;
movieOpt.T4V = T4V;
%% drawing source and target
title_str = ['Img$(f_0)$ (' dist_name ')'];
if movieOpt.print_dist
    title_str = [title_str, '  dist=' num2str(tot_dist)];
end

if options.draw_mesh
    if ~strcmp(dist_draw,'None')
        clf;
    end
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
    if mov_i <= movieOpt.max_frame 
        getframe(gcf);
        mov_i= mov_i+1;
    end
end

%% 
perimiter_norm = norm(const_mesh.perimiter) ;
is_all_converged=false;
continue_optimization  =1;
out_iter =1;
while continue_optimization
    for opt_index =1:length(optParam_list) %iterating over measures
        optParamH = optParam_list{opt_index};
        if optParamH.is_stopped  
            continue;            
        end
        optParam  = struct(optParamH);
        opLS = OptimizationParams.SubStruct(optParam,'LS');
        opLS.penalize_flips_t = false(0,1);
        if run_mex 
            E_dist=dist_gradient_analyticMex_mex(const_mesh,var_mesh.fV,Loc2GlobV,T4V,opLS); 
        else
            E_dist=dist_gradient_analyticMex(const_mesh,var_mesh.fV,Loc2GlobV,T4V,opLS); 
        end
        E_dist = E_dist/mesh.WcSum;
        if (optParamH.current_iter == 1) %intialize distortions 
            optParamH.InitDistortion(E_dist);
        end
        if optParamH.Check4Stopping(E_dist) %update current opt.status
            continue;
        end

        for inner_i = 1:optParamH.cycle_num %iterations of the  current measure
            disp(['iteration ' num2str(out_iter) ' __________________________________________']);
            disp('before>>');
            optParamH.print_status();
            if inner_i > 1
               optParam  = struct(optParamH); 
            end
            optParam.block_max_iter = optParamH.GetMaxBlockIter();
            disp('after>>');
            tic;
                if inner_i==1 
                    results.distS =[];
                end
            if optParam.run_mexed_solver 
                [var_mesh,results]=ABCD_single_measure_wrapper(const_mesh,var_mesh,meshData,optParam,results.distS);   
            else
               [var_mesh,results]=GD_BCQN_coordinate_descent_solvers(const_mesh,var_mesh,T4V,V4V, Loc2GlobV,optParam,results.distS);   
            end
            if options.draw_blocks && isfield(options,'draw_block_fig_name')
                block_partion_fig = [options.draw_block_fig_name optParamH.get_string_descripor() '_iter' num2str(out_iter) '.fig'];
                block_num_str = num2str(results.block_num(end));
                title([optParamH.get_string_descripor() ' iter' num2str(out_iter) ' ' block_num_str ' blocks']);
                savefig(block_partion_fig);
            end
            
            out_iter = out_iter+optParamH.iter_per_cycle;
            optParamH.AddIterData(results.dist(end),results.delta_energy,results.block_num(end),...
                                   results.flip_num(end)+results.degen_num(end));
            %% Drawing 
            if options.draw_mesh && ~isempty(optParam.dist_draw)
                mesh.fV    = var_mesh.fV; 
                mesh.rad_v = var_mesh.rad_v;
                mesh.dis_v = var_mesh.dis_v;
                mesh.dis_t = var_mesh.dis_t;
                mesh.SV_t  = var_mesh.SV_t;
                
                %check if the last results in distortion measurements are valid for drawing
                if ~strcmp(optParam.dist_name,dist_draw) || ~isfield(results.distS,'v')
                    opLS=OptimizationParams.SubStruct(optParam_list{optimizer_index},'LS');
                    opLS.penalize_flips_t = false(0,1);
                    is_fixed_v = const_mesh.is_fixed_v;
                    const_mesh.is_fixed_v(:)=0;
                    if run_mex
                        [~,grad,results_distS]=dist_gradient_analyticMex_mex(const_mesh,mesh.fV,Loc2GlobV,T4V,opLS );
                    else
                        [~,grad,results_distS]=    dist_gradient_analyticMex(const_mesh,mesh.fV,Loc2GlobV,T4V,opLS );
                    end
                    const_mesh.is_fixed_v = is_fixed_v;
                    results_distS.grad = grad; 
                else
                     results_distS =  results.distS;
                end
                
                clf;
                if d==4
                    draw_opt_results_3D_movie_analytic(mesh,mesh.fV,dist_draw,results_distS, movieOpt);
                else
                    draw_opt_results_Surf_analytic(mesh,mesh.fV,dist_draw,results_distS, movieOpt);
                end
                if isfield(movieOpt,'draw_rec') && length(movieOpt.draw_rec) > 1
                    hold on;
                    draw_bbox_3D(movieOpt.draw_rec(1,:),movieOpt.draw_rec(2,:),movieOpt.rec_LineSpec);
                end
                
                if movieOpt.single_colmap
                    caxis manual; caxis([col_min col_max]);
                end
                if options.draw_colbar; colorbar; end
                title_str=['Img$(f_{' num2str(out_iter) '})$ (' dist_name ')'];
                if movieOpt.print_dist
                    title_str = [title_str, '  tot dist=' num2str(tot_dist)];
                end
                title(title_str,'FontSize',16,'interpreter','latex');
                hold on
               
                if mov_i <= movieOpt.max_frame 
                    getframe(gcf);
                    mov_i= mov_i+1;
                end
            end
            
            if optParamH.Check4Stopping(results.dist(end)) 
               break; %go to the next distortion, this one have been optimized 
            end
            %checking for final convergence 
            if options.check_final_convergence
                grad_norm =norm(results.distS.grad);
                char_grad_criteria = (grad_norm < options.char_grad_threshold * perimiter_norm);
                disp_norm_criteria = (results.delta_energy(end) < options.disp_norm_threshold);
                is_valid_map = (results.flip_num(end) ==0) && (results.degen_num(end) ==0);
                is_all_converged = char_grad_criteria && disp_norm_criteria && is_valid_map && opt_index == optimizer_i;
                if is_all_converged
                    disp(['Convergence criteria is met on iteration' num2str(out_iter) newline ,...
                        '   --->Thresholds: perimiter norm=' num2str(options.char_grad_threshold) ', Disp Norm=' num2str(options.disp_norm_threshold)]);
                end
                if opt_index ==1; is_all_converged =0; end 
                    
            end
            
            if out_iter > options.iter_num || is_all_converged
                break; 
            end
        end
        if out_iter > options.iter_num
           break;
        end
    end
    continue_optimization = (out_iter < options.iter_num ) && (~is_all_converged) ; 
end
fV= mesh.fV;
end

