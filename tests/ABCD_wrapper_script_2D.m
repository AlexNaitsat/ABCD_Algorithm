%%  Copyright @2019. All rights reserved.
%  Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

close all
root_path =pwd;
Set_ABCD_Paths;
%% setting some maltab global params for text
disp(' - Loading parameteres to set the problem');
set(groot,'defaulttextinterpreter','latex');
postfix = []; 

%%loading source/ in mesh
[in_path,in_name,ext]=fileparts(in_mesh);

if strcmp(ext,'.obj')
    %[F,V,extra]=read_obj(in_mesh);
    [V,F,~] = load_obj_mesh([in_path '/' in_name]);
elseif  strcmp(ext,'.mat')
    load(in_mesh);
    if ~exist('F','var') &&  exist('T','var')
        F =T;
    end
end


n= length(V);
load(tmesh_file);
if exist('I_fixed','var')
   options.fixed_vertices = I_fixed;
   
end

%% some drawing of source mesh
figure;
trisurf(F,V(:,1),V(:,2),V(:,3));
hold on;

view(movieOpt.view);
axis equal;
title('Source');
op.null =0;

%% Generating/loading target mesh
    [in_path,target_name,ext]=fileparts(target_mesh);

    if strcmp(ext,'.obj')
        [~,fV,~]=read_obj(target_mesh);
    elseif strcmp(ext,'.mat')
        load(target_mesh);
        if exist('mesh','var') && isfield(mesh,'fV')
            fV= mesh.fV;
        end
        [fVrows,fVcolumns]=size(fV);
        if (fVcolumns<3)
            fV = [fV zeros(fVrows,3-fVcolumns)];
        end
    end
if ~isfield(options,'fig_name')
    if exist('target_name','var') &&  ~isempty(target_name)
        options.fig_name = target_name;
    else
        options.fig_name =in_name;
    end
end

if isfield(options,'target_pos_normal') %setting positive orientation
    tmesh.target_pos_normal  = options.target_pos_normal;
else
    tmesh.target_pos_normal =[0 0 1];
end
if ~isfield(options,'visualize_iterations')
    options.visualize_iterations = 0;
end

options.fig_name= [options.fig_name postfix];
%%
if isfield(options,'planar_target') && options.planar_target 
    fV=fV(:,1:2); 
end
if options.visualize_iterations
    disp( ['-I- This is the Eigen version that reports and visualzes results for each iteration' newline ...
        '    To get a faster performance: disable visualization and reports, set parallel flags and use pardiso solver' newline ...
        '    (flags draw_mesh, parallel_mode and use_pardiso)']);
    promp = ' ----- Press enter to continue ----';
    y = input(promp);
    
    fV  = UnifiedDistortionsOptimizationIterVisualize(tmesh,fV,optParamList,options,movieOpt);
else
    disp( ['-I- This is the Eigen version' newline ...
        'To get a faster performance:  set parallel flags and use pardiso solver' newline ...
        '    (flags parallel_mode, parallel_grad, parallel_hessian, parallel_energy, use_pardiso)']);
    promp = ' ----- Press enter to continue ----';
    y = input(promp);
    
    fV  = UnifiedDistortionsOptimization(tmesh,fV,optParamList,options,movieOpt);
    
end