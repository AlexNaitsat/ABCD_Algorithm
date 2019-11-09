%%  ABCD(PN) grid deformation
addpath('tests');
warning off
clear all 
close all 
in_mesh       =  './data/bar100x100sub.obj';
target_mesh   =  './data/bar100x100sub_noisy.obj';
tmesh_file    =  './data/tmesh_bar100x100sub.mat';

load('./data/visualization_params.mat');
load('./data/general_params.mat');

%energy/solver/termination-criteria specifications 
optParamList = {...
    OptimizationParams(OptimizationParams.ParseOptions(...
          'LS_iter',100,'dist_name','FS','dist_draw','FS','cycle_num',4,...
          'solver','GD','LS_interval',1.05,'penalize_flips',0,'Lambda',0,...
          'block_max_iter',[1 10])...
    ),...
    OptimizationParams(OptimizationParams.ParseOptions('LS_iter',50,'LS_interval',0.5,'LS_bt_step',0.8,...
          'project_kernel',1,'dist_name','SymDirSigned2D','dist_draw','SymDir2D','cycle_num',4,'local_global_policy',4,...
          'solver','PN','block_max_iter',[1 4])...  
    )...
};

options.fig_name= './reports/grid_deformation';
options.iter_num =15;
options.dist_name = 'SymDirSigned2D';
options.dist_draw = 'SymDirSigned2D';
options.free_bnd =1;

run ABCD_wrapper_script_2D.m
%% ABCD(PN) elephant with random interior
clear all 
close all 
ENABLE_DRAWING=1; 
in_mesh       =  './data/elephant120K_s.obj';
target_mesh   =  './data/elephant120K_t_1.mat';
tmesh_file    = './data/tmesh_of_elephant_120K.mat';

load('./data/visualization_params.mat');
load('./data/general_params.mat');
%energy/solver/termination-criteria specifications 
optParamList = {...
        OptimizationParams(OptimizationParams.ParseOptions(...
        'LS_iter',1000,'dist_name','FS','dist_draw','FS','cycle_num',4,...
        'solver','GD','LS_interval',2,'penalize_flips',0,'Lambda',[0 0],'LS_alpha',10^-5,...
        'block_max_iter',[1 10])...
        ),...
        OptimizationParams(OptimizationParams.ParseOptions('LS_iter',500,'LS_interval',0.5,'LS_bt_step',0.8,...
        'project_kernel',1,'dist_name','SymDirSigned2D','dist_draw','SymDir2D','cycle_num',4,...
        'solver','PN','block_max_iter',[1 2])...  
        )...
};

options.fig_name= './reports/elephant_high';
options.iter_num =30;
options.dist_name = 'SymDirSigned2D';
options.dist_draw = 'SymDirSigned2D';
options.free_bnd =0;

run ABCD_wrapper_script_2D.m
%% ABCD(PN) octopus constrained parametrization
clear all 
close all 

in_mesh       =  './data/octopus_s.obj'; 
target_mesh   =  './data/octopus_t_1.mat';
tmesh_file   =  './data/tmesh_of_octopus.mat';

load('./data/visualization_params.mat');
load('./data/general_params.mat');
movieOpt.mark_fixed_vertices=1;
options.is_uv_mesh=1;

optParamList = {...
        OptimizationParams(OptimizationParams.ParseOptions(...
        'LS_iter',100,'dist_name','FS','dist_draw','FS','cycle_num',4,...
        'solver','GD','LS_interval',1.05,'penalize_flips',0,'Lambda',[0 0],'LS_alpha',10^-5,...
        'block_max_iter',[1 10])...
        ),...
        OptimizationParams(OptimizationParams.ParseOptions('LS_iter',100,'LS_interval',0.5,...
        'dist_name','SymDirSigned2D','dist_draw','SymDir2D','cycle_num',4,...
        'solver','PN','block_max_iter',[1 4])...  
        )...
};
options.fig_name= './reports/octopus';
options.iter_num =30;
options.dist_name = 'SymDirSigned2D';
options.dist_draw = 'SymDirSigned2D';
options.free_bnd =1;

run ABCD_wrapper_script_2D.m