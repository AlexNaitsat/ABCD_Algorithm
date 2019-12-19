%% Copyright @2019. All rights reserved.
%  Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)
%%--------------------------------------------------------------------
% This is the Problem class for managing  parameters  of each individual
% distortion measure, including peformance estiamtes and parameters that control
% blending  for computing  block  partitioning thresholds.
% Some member variables are only for internal use.
%%
classdef OptimizationParams < handle
    
    properties
        dist_name='SymDir2D';           %distortion function name
        dist_draw='';      %distortion function for visualization
        is_recorded=1;     %enable recording of distortion values and related parameters
        
        %% constant iteration params
        max_iter_num;    %maximal number of total iterations
        cycle_num;       %number of epoch cycles and epoch iterations
        epoch_num = inf; 
        iter_in_current_epoch=0; %internal params that count iterations
        iter_per_cycle=1;  
        
        auto_epoch_reset =1; 
        cycles4auto_reset =2;
        fixer_auto_epoch_reset=0; 
        
        
        %% Varying iteration params
        current_iter;    %current iteration number
        LG_cycle         %number of current  local-global cycles
        first_LG_cycle =1; 
        
        %%
        project_kernel=0;     %project out kernel (rigid transform. directions)
        solver =  'PN';       %core-solver name
        solver_data;  
        use_pardiso = 0;  %pardiso need to be installed
        %%
        block_threshold; 
        block_max_iter=10;
        %%
        is_local_global=1; 
        is_global=0;       
        local_global_policy =6 ;
        disp_blend  =0.3;       %for performance evaluation 
        %% line search
        LS_interval =0.95;%interval length 
        LS_iter  =300;    %Maximal number of line search iterations
        LS_bt_step =0.9;  %back tracking line search steps
        LS_alpha = 1e-5; 
        %%
        stop_criteria={'<=',0,1}; %termination criteria (char. grad, char. grad + displacement norm and etc.)
        
        
        is_stopped=0;
        distLoc;  %differences in distortions values 
        distGlob;
        displGlob; %displacement energy
        displLoc;
        blockNumGlob;
        blockNumLoc;
        
        glob_loc_performance_ratio=1;
        thresholdLoc;    %partitioning thresholds
        thresholdGlob;
        isLocalStage;
        isFirstLocal=0; %1st titeration is a global step 
        dist4iters;     %list of total distortions per iterations
        invalid4iters;  %invalid element number 
        threshold4iter; 
        %%
        EPS=10^-7; 
        eps4epoch =10^-8; %threshold on slow progress 
        dis_eps = 10^-3; 
        sing_eps = 10^-5; %singularity threshold
        Lambda = [0 0];   %penalty for map fixer measures 
        max_ratio    = inf;
        check_cycles_num=2;
        zero_grad_eps = 10^-16; 
        
        %%
        penalize_flips=1; %Enable/disable  flip barrier 
        penalize_inf = 0;
        check_flips =1; 
        
        skip_fixed_vertices =1;
        penalize_cell_flips =1;
        penalize_flips_t; 
        singular_t; 
        %%
        boundary_gradient_threshold = 10^-4; %used for visualization only
        performance4LG_iter;
        %% mexed solver params
        run_mexed_solver =1;
        max_global_iterations =1;
        parallel_mode  = 1;
    end
    
    methods
        function o = OptimizationParams(options, init_dist,init_invalid)
            %% optimization structures
            if isfield(options,'is_local_global'); o.is_local_global = options.is_local_global; end
            if isfield(options,'is_global'); o.is_global = options.is_global; end
            
            %%
            if isfield(options,'project_kernel'); o.project_kernel = options.project_kernel; end;
            if isfield(options,'solver'); o.solver = options.solver; end
            if isfield(options,'solver_data'); o.solver_data = options.solver_data; end
            
            %% iterations
            if ~isfield(options,'max_iter_num'); options.max_iter_num =10; end
            o.max_iter_num = options.max_iter_num;
            if ~isfield(options,'check_cycles_num'); options.check_cycles_num =3; end
            o.check_cycles_num =options.check_cycles_num;
            if isfield(options,'block_max_iter');  o.block_max_iter =options.block_max_iter; end
            
            if ~isfield(options,'cycle_num'); options.cycle_num =ceil(max_iter_num*0.1); end
            o.cycle_num =options.cycle_num;
            if ~isfield(options,'iter_per_cycle'); o.iter_per_cycle = options.iter_per_cycle; end
            if isfield(options,'epoch_num'); o.epoch_num = options.epoch_num; end
            
            if isfield(options,'fixer_auto_epoch_reset'); o.fixer_auto_epoch_reset = options.fixer_auto_epoch_reset; end
            
            %% barriers
            if isfield(options,'penalize_flips'); o.penalize_flips = options.penalize_flips; end
            if isfield(options,'check_flips'); o.check_flips = options.check_flips; end;
            
            %% Line search params
            if isfield(options,'LS_interval'); o.LS_interval =options.LS_interval; end %interval length 
            if isfield(options,'LS_iter'); o.LS_iter = options.LS_iter; end       
            if isfield(options,'LS_bt_step'); o.LS_bt_step = options.LS_bt_step; end %back tracking step size
            if isfield(options,'LS_alpha'); o.LS_alpha = options.LS_alpha; end 
            %% Gradient params
            if isfield(options,'boundary_gradient_threshold') %used only for visualization
                o.boundary_gradient_threshold=options.boundary_gradient_threshold;
            end
            
            %% policies
            if isfield(options,'local_global_policy'); o.local_global_policy = options.local_global_policy; end %interval length relative to ring radius
            if isfield(options,'disp_blend');          o.disp_blend = options.disp_blend;   end 
            %% intializing arrays
            o.distLoc =zeros(o.max_iter_num,1);
            o.distGlob=zeros(o.max_iter_num,1);
            o.displGlob=zeros(o.max_iter_num,1);
            o.displLoc =zeros(o.max_iter_num,1);
            
            o.blockNumGlob = zeros(o.max_iter_num,1);
            o.blockNumLoc  = zeros(o.max_iter_num,1);
            
            o.dist4iters=zeros(o.max_iter_num,1);
            o.invalid4iters=zeros(o.max_iter_num,1);
            
            o.threshold4iter = -ones(o.max_iter_num,1);
            o.performance4LG_iter = zeros(o.max_iter_num,1);
            
            %initial point params
            o.current_iter=1;
            if isfield(options,'isFirstLocal'); o.isFirstLocal = options.isFirstLocal; end 
            
            o.isLocalStage = o.isFirstLocal;
            if o.local_global_policy ==5 || o.local_global_policy ==6 %constant threshold=1 (purely, global, or global without blending)
                o.thresholdLoc=1;
                o.thresholdGlob=1;
            elseif o.local_global_policy ==7 
                o.thresholdLoc=0;
                o.thresholdGlob=0;
            else
                o.thresholdLoc=0;   %partitioning thresholds
                o.thresholdGlob=1; 
            end
            if o.isFirstLocal
                o.block_threshold =  o.thresholdLoc;  %starts from Local  stage
            else
                o.block_threshold =  o.thresholdGlob; %starts from Global stage
            end
            o.LG_cycle =1;
            
            %% distortion params
            if isfield(options,'dist_name'); o.dist_name = options.dist_name;  end
            
            %% additional params of map fixer distortions
            if isfield(options,'Lambda'); o.Lambda = options.Lambda;  end
            if length(o.Lambda)==1
                o.Lambda  =[o.Lambda o.Lambda];
            end
            
            if ~exist('init_dist','var'); init_dist =0; end
            o.dist4iters(o.current_iter)=init_dist;
            
            if ~exist('init_invalid','var'); init_invalid =0; end
            o.invalid4iters(o.current_iter)=init_invalid;
            
            o.threshold4iter(o.current_iter)=o.block_threshold;
            
            if ~exist('dist_draw','var'); o.dist_draw  = options.dist_draw; end
            
            
        end
        %%
        function InitDistortion(o,init_dist) 
            o.dist4iters(1)=init_dist;
            o.block_threshold(1) = o.block_threshold;
        end
        
        function AddIterData(o,dist, displ,block_num,invalid_num)
            o.current_iter = o.current_iter+1;
            o.iter_in_current_epoch = o.iter_in_current_epoch+1;
            o.dist4iters(o.current_iter)= dist;
            
            o.invalid4iters(o.current_iter) = invalid_num;
            
            if invalid_num ==0 && o.local_global_policy == 4
                o.local_global_policy  = 5;
            end
            
            if o.is_local_global 
                if o.isLocalStage
                    o.distLoc(o.LG_cycle) = max([o.dist4iters(o.current_iter-1)-... 
                        o.dist4iters(o.current_iter), o.EPS]);
                    
                    o.displLoc(o.LG_cycle) = displ +o.EPS;
                    o.blockNumLoc(o.LG_cycle)= block_num;
                else
                    o.distGlob(o.LG_cycle) = max([o.dist4iters(o.current_iter-1)-...
                        o.dist4iters(o.current_iter), o.EPS]); 
                    
                    o.displGlob(o.LG_cycle) = displ + o.EPS;
                    o.blockNumGlob(o.LG_cycle)= block_num;
                end
                
                
                if o.isFirstLocal ~= o.isLocalStage %Check for the end of the current cycle
                    %Updating Local vs Global performance ratio
                    LG_start = max(o.first_LG_cycle,o.LG_cycle-o.check_cycles_num+1); %set first index for performance evaluation
                    
                    %performance evaluation over the passed global-local cycle.
                    cycle_indices = LG_start:o.LG_cycle; %indices of relevant iterations
                    gl_ratios = o.distGlob(cycle_indices)./o.distLoc(cycle_indices);
                    gl_displ_ratios = o.displGlob(cycle_indices)./o.displLoc(cycle_indices);
                    cycle_len = length(cycle_indices);
                    
                    % cycles where  local and global stages has the same number of blocks
                    same_block_num =  o.blockNumGlob(cycle_indices) == o.blockNumLoc(cycle_indices);
                    gl_ratios(same_block_num) =1;      
                    gl_displ_ratios(same_block_num)=1; 

                    rScaleFunc =@(x,i) x;
                    if o.isFirstLocal
                        glob_indices =  2*(1:cycle_len);
                        loc_indices  =  2*(1:cycle_len) -1;
                    else
                        loc_indices  =  2*(1:cycle_len);
                        glob_indices =  2*(1:cycle_len) -1;
                    end
                    
                    delta_glob = o.distGlob(cycle_indices)./o.dist4iters(glob_indices);
                    delta_loc  = o.distLoc(cycle_indices)./o.dist4iters(loc_indices);
                    
                    delta_ratios = delta_glob./delta_loc;
                    delta_ratios(same_block_num)=1; %reset ratios
                    scaled_sum=0;
                    iter_vec = LG_start:o.LG_cycle;
                    for ii = 1:cycle_len
                        scaled_sum = scaled_sum + rScaleFunc(delta_ratios(ii)+ o.disp_blend*gl_displ_ratios(ii),cycle_indices(ii));
                    end
                    o.glob_loc_performance_ratio = scaled_sum /(cycle_len*(1+o.disp_blend*1)); %if delta_ratios,gl_displ_ratios==1
                    o.performance4LG_iter(o.LG_cycle) = o.glob_loc_performance_ratio;
                    
                    thGlob= o.thresholdGlob; thLoc = o.thresholdLoc;
                    %% local global blending policies
                    if o.local_global_policy == 3
                        scaleFunc = @(x) x;
                        if o.glob_loc_performance_ratio>=1 
                            
                            o.thresholdLoc  = thGlob - (thGlob-thLoc)/scaleFunc(o.glob_loc_performance_ratio);
                            o.thresholdGlob =1-(1-thGlob)/scaleFunc(o.glob_loc_performance_ratio); %move global threshold rightword by same distance as local threshold
                        else
                            o.thresholdGlob  = thLoc + (thGlob-thLoc)*scaleFunc(o.glob_loc_performance_ratio);
                            o.thresholdLoc   = thLoc*scaleFunc(o.glob_loc_performance_ratio);
                        end
                        
                    elseif o.local_global_policy > 3
                        o.thresholdLoc  = 1;  
                        o.thresholdGlob = 1;  
                    end
                    
                    o.LG_cycle = o.LG_cycle+1; %counting local-global cycles
                    
                    to_reset_params = 0;
                    if o.iter_in_current_epoch > o.epoch_num 
                        disp('--- Starting new epoch  ---');
                        to_reset_params=1;
                    elseif o.auto_epoch_reset && o.iter_in_current_epoch >= 2*o.cycles4auto_reset
                        iter_indices = o.LG_cycle-o.cycles4auto_reset:o.LG_cycle-1; 
                        loc_tot_diff = sum(o.distLoc(iter_indices)  + o.disp_blend*o.displLoc(iter_indices) );
                        glob_tot_diff =sum(o.distGlob(iter_indices)+ o.disp_blend*o.displGlob(iter_indices) );

                        if (loc_tot_diff + glob_tot_diff)/(2*o.cycles4auto_reset) < o.eps4epoch 
                            disp('--- Epoch is automatically reset due to a slow progress ---');
                            to_reset_params=1;
                        end
                    end

                    if   ~to_reset_params && o.fixer_auto_epoch_reset && o.iter_in_current_epoch >= o.fixer_auto_epoch_reset%2*o.cycles4auto_reset
                        iter_indices = (o.current_iter-o.fixer_auto_epoch_reset): o.current_iter;
                        if o.invalid4iters(iter_indices(end)) >= o.invalid4iters(iter_indices(1))
                            to_reset_params=1;
                            disp(['--- Epoch is automatically reset due to a slow FIXER progress ---' newline]);
                        end
                    end
                    
                    if to_reset_params 
                        o.thresholdLoc  = 0;               
                        o.thresholdGlob = 1;               
                        o.iter_in_current_epoch=0;
                        o.first_LG_cycle =  o.LG_cycle; 
                    end
                end
                
                o.isLocalStage = ~o.isLocalStage; %switch between Local and Global stages
                if o.isLocalStage
                    o.block_threshold =  o.thresholdLoc;
                else
                    o.block_threshold =  o.thresholdGlob;
                end
            else 
                o.LG_cycle = o.LG_cycle+1;
                o.distGlob(o.LG_cycle) = max([o.dist4iters(o.current_iter-1)-...
                    o.dist4iters(o.current_iter), o.EPS]); 
                
            end
            
            Check4Stopping(o,dist);%Check for termination 
            o.threshold4iter(o.current_iter)= o.block_threshold;
        end
        
        function is_stopped = Check4Stopping(o,dist)
            if strcmp(o.stop_criteria{1},'<=')
                is_stopped = dist <= o.stop_criteria{2};
            else
                is_stopped = 0;
            end
            o.is_stopped = is_stopped;
            if strcmp(o.dist_name,'FS') && o.invalid4iters(o.current_iter) > 0
                o.is_stopped = 0; 
            end                  
            
        end
        
        %%
        function print_status(o)
            if o.isLocalStage
                mode_str = 'Local';
            else
                mode_str = 'Global';
            end
            
            disp(['Total ',o.dist_name, '=',num2str(o.dist4iters(o.current_iter))]);
            %disp(['Total ',o.dist_name, '=',num2str(o.dist4iters(o.current_iter)),...
            %    ',', mode_str, ' opt. threshold=',num2str(o.block_threshold), ', glob/loc=', num2str(o.glob_loc_performance_ratio)]);
            disp(['current iter=',num2str(o.current_iter),' LG cycle=', num2str(o.LG_cycle)]);
            
        end
       
        function blockIter =   GetMaxBlockIter(o)
            if length(o.block_max_iter) ==1
                blockIter =o.block_max_iter; %constant block iterations
            else %varying block iterations
                blockIter = min([o.block_max_iter(1)+ o.LG_cycle-1,o.block_max_iter(2)]); %local steps start with  max_block_iter=1
            end
            
        end
        
    end
    %%
    
    
    methods(Static)
        function opt= ParseOptions(varargin)
            p = inputParser;
            isBool =@(x) (x==0 ||  x==1);
            
            addOptional(p,'dist_name','K2D');
            addOptional(p,'dist_draw','K2D');
            addOptional(p,'max_iter_num',1000);
            addOptional(p,'iter_per_cycle',1);
            addOptional(p,'block_max_iter',10);
            addOptional(p,'cycle_num',10);
            addOptional(p,'epoch_num',inf);
            
            addOptional(p,'fixer_auto_epoch_reset',0);
            addOptional(p,'is_local_global',1,isBool);
            addOptional(p,'penalize_flips',1,isBool);
            addOptional(p,'check_flips',1,isBool);
            addOptional(p,'solver','PN');
            addOptional(p,'solver_data',struct());
            
            
            addOptional(p,'is_global',0,isBool);
            addOptional(p,'LS_interval',0.8);
            addOptional(p,'LS_iter',300);
            addOptional(p,'LS_bt_step',0.9);
            addOptional(p,'LS_alpha',1e-5);
            addOptional(p,'Lambda',1);
            addOptional(p,'project_kernel',0);          
            addOptional(p,'local_global_policy',6);
            addOptional(p,'isFirstLocal',0);
            addOptional(p,'disp_blend',0.3); 
            
            parse(p,varargin{:});
            opt=p.Results;
        end
        
        %% get  sub structures for coder.
        function s =SubStruct(o,sub_str)
            switch  sub_str
                case 'LS' %line search fields
                    s = struct('dist_name', o.dist_name ,'sing_eps',o.sing_eps,'Lambda',o.Lambda,... %distortion values
                        'boundary_gradient_threshold', o.boundary_gradient_threshold, 'skip_fixed_vertices',o.skip_fixed_vertices,...     %and gradients
                        'penalize_cell_flips', o.penalize_cell_flips,'penalize_flips_t', o.penalize_flips_t,'check_flips',o.check_flips,... %flips
                        'penalize_inf',o.penalize_inf, 'penalize_flips', o.penalize_flips,'max_ratio',o.max_ratio, ... %'singular_t', o.singular_t,... %singularitites (chech if singular_t is used, its nto field of this object)
                        'LS_iter', o.LS_iter,'LS_alpha',o.LS_alpha, 'LS_bt_step',o.LS_bt_step, 'LS_interval',o.LS_interval... %core line-search
                        );
                case 'block' %for block partitioning
                    s = struct('zero_grad_eps', o.zero_grad_eps ,'block_threshold',o.block_threshold);
                case 'GD' %GD solver params
                    s = struct('dist_name',o.dist_name,'block_max_iter', o.block_max_iter , 'dis_eps',o.dis_eps,'penalize_flips', o.penalize_flips,'LS_interval',o.LS_interval);
                    s.stop_criteria = o.stop_criteria; 
                otherwise
                    error(['No such  substructure ' sub_str]);
                    
            end
            
        end
    end
end

