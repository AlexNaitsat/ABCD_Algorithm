%%  Copyright @2020. All rights reserved.
% Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs for wrapper script:
% in_mesh       : source mesh file; 'obj', or 'mat' with saved variables 
%                 T (triangle connectivity), V (source vertex coordinates)
% target_mesh   : target intialization; 'obj', or 'mat' with saved variable
%                 fV (target vertex coordinates), or 'uv' to initialize it
%                 by UV coordinates from 'in_mesh' file
% options       : constraints, dimensions of source
%                 mesh and draw_mesh flag
% movieOpt      : additional options for mesh visualization
% optimizer_spec: struct with specifications for the optimizer solver
% fixer_spec    : struct with specifications for the fixer solver
%--------------------------------------------------------------------------
% Output of wrapper script:
% fV         : final target coordinates of vertices
% tmesh      : mesh data structure 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initializing common parameters
clear all 
close all
Set_ABCD_Paths
load('./data/general_options.mat');
load('./data/fixer_optimizer_spec.mat');
optimizer_spec.use_pardiso =0;%<--Set Eigen or Pardiso
                              %To use pardiso compile mex with _USE_PARDISO                       
%% Planar deformations
in_mesh     =  './data/elephant_2K_s.obj';
target_mesh =  './data/elephant_2K_t.obj';
options.free_bnd       = 0;       
options.fixed_vertices = [];
options.is_uv_mesh     = 0; 
run ABCD_MexWrapperScript.m
%%
in_mesh     =  './data/elephant_30K_s.obj';
target_mesh =  './data/elephant_30K_t.mat';
options.free_bnd       = 0;       
options.fixed_vertices = [];
options.is_uv_mesh     = 0; 
run ABCD_MexWrapperScript.m
%%
in_mesh     =  './data/elephant_120K_s.obj';
target_mesh =  './data/elephant_120K_t.mat';
options.free_bnd       = 0; %fixed boundary 
options.fixed_vertices = [];%no additional anchors 
options.is_uv_mesh     = 0; %planar mesh
run ABCD_MexWrapperScript.m

%% Constrainted parametrizations
in_mesh     =  './data/octupus_6K.obj';
target_mesh =  './data/octupus_6K_initialization.mat';
options.free_bnd       = 1; %boundary is free except the acnhors
options.fixed_vertices = [1266 2097 3080 3320 2280 1770 2706 3641];%anchors
options.is_uv_mesh     = 1;%surface mesh
run ABCD_MexWrapperScript.m
%%
in_mesh     =  './data/octupus_24K_s.obj';
target_mesh =  './data/octupus_24K_t.mat';
options.free_bnd       = 1; %boundary is free except the acnhors
options.fixed_vertices = [1266 2097 3080 3320 2280 1770 2706 3641];
options.is_uv_mesh     = 1;
run ABCD_MexWrapperScript.m
%%
in_mesh     =  './data/octupus_100K_s.obj';
target_mesh =  './data/octupus_100K_t.mat';
options.free_bnd       = 1; 
options.fixed_vertices = [1266 2097 3080 3320 2280  1770  2706  3641];
options.is_uv_mesh     = 1;
run ABCD_MexWrapperScript.m

%% BFF intialization schemes
in_mesh     = './data/D1_01121_bff.obj';
target_mesh = 'uv';       %initialized by texture UV coordinates
options.free_bnd       = 1;  % 
options.fixed_vertices = []; %unconstrained
options.is_uv_mesh     = 1;
run ABCD_MexWrapperScript.m

%% 
in_mesh     = './data/D1_02392_bff_dinobird.obj';
target_mesh = 'uv';
options.free_bnd       = 1;
options.fixed_vertices = [];
options.is_uv_mesh     = 1;
run ABCD_MexWrapperScript.m

%% 
in_mesh     = './data/D1_00478_bff.obj';
target_mesh = 'uv';
options.free_bnd       = 1;
options.fixed_vertices = [];
options.is_uv_mesh     = 1;
run ABCD_MexWrapperScript.m

%% Intialization with  decimated UV-map
in_mesh     = './data/kingkong09.obj';
target_mesh = './data/kingkong09_UV_20_flips.mat';
options.free_bnd       = 1;  %
options.fixed_vertices = []; %unconstrained
options.is_uv_mesh     = 1;
run ABCD_MexWrapperScript.m

%% Local-global blending with precomputed partitioning threshold
in_mesh     =  './data/elephant_30K_s.obj';
target_mesh =  './data/elephant_30K_t.mat';
options.free_bnd       = 0; 
options.fixed_vertices = [];
options.is_uv_mesh     = 0; 

load('./data/elephant_30K_partition_thresholds.mat');
fixer_spec.K_hat_list     = fixer_threshold;
fixer_spec.K_hat_size     = length(fixer_threshold);                     
optimizer_spec.K_hat_list = optimizer_threshold;
optimizer_spec.K_hat_size = length(optimizer_threshold);                     

run ABCD_MexWrapperScript.m
%clear thresholds
optimizer_spec.K_hat_list = []; optimizer_spec.K_hat_size = 0;
fixer_spec.K_hat_list     = []; fixer_spec.K_hat_size     = 0;

%% 3D problems
in_mesh     = './data/wrench_bended_3d_s.mat';
target_mesh = './data/wrench_bended_3d_t.mat';

options.free_bnd         = 1;
load('./data/wrench_bended_3d_anchors.mat');
options.fixed_vertices   = wrnech_bended_anchors;

options.draw_mesh         = 0;%tetramesh drawing is slow!
run ABCD_MexWrapperScript.m

%% 
in_mesh     = './data/twisted_3d_bar_30K_s.mat';
target_mesh = './data/twisted_3d_bar_30K_t.mat';

load('./data/bar_30K_anchors.mat');
options.free_bnd         = 1;
options.fixed_vertices   = bar_30K_anchors;
options.draw_mesh        = 0;
run ABCD_MexWrapperScript.m
%% 
in_mesh     = './data/twisted_3d_bar_12K_s.mat';
target_mesh = './data/twisted_3d_bar_12K_t.mat';

options.free_bnd         = 1;
options.fixed_vertices   = [];%unconstrained
options.draw_mesh        = 1;
run ABCD_MexWrapperScript.m
