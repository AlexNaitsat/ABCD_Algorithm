function Set_ABCD_Paths(root_path)
% %% seeting  root paths
if ~exist('root_path','var')
    root_path= [pwd '\'];
end
  
%% adding patches for matlab code
addpath(root_path);
addpath(genpath([root_path 'Coder']));
addpath([root_path 'tests']);
addpath(genpath([root_path '\utils']));
addpath(genpath([root_path '\visualization']));
addpath(genpath(['.\mexed_solver']));
addpath(root_path);

%% setting some maltab global params related to how  text is displayed
set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
end