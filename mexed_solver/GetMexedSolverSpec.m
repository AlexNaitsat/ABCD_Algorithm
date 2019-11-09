function optimization_spec = GetMexedSolverSpec(op)

%default values
invalid_penalty= zeros(3,1);
invalid_penalty(1:length(op.Lambda)) = op.Lambda; 
optimization_spec = struct('solver_num',1,...
                        'energy_num',0,...
                        'invalid_penalty',invalid_penalty,...
                        'max_block_iterations',op.block_max_iter, ...
                        'max_global_iterations',op.max_global_iterations,...
                        'spd_threshold',1e-6,...
                        'is_signed_svd',1,...
                        'is_parallel', 1, ...
                        'ls_interval',op.LS_interval,...
                        'ls_alpha',op.LS_alpha,...
                        'ls_beta',op.LS_bt_step,...
                        'is_flip_barrier', op.penalize_flips,...
                        'ls_max_iter', op.LS_iter);
                     
switch op.dist_name
    case 'ARAP_Signed'
        optimization_spec.energy_num =0;
        optimization_spec.is_signed_svd =1;
        optimization_spec.constant_step_size =1.0;
    case 'SymDir2D'
        optimization_spec.energy_num =1; 
        optimization_spec.is_signed_svd =1;
     case 'FS'
        optimization_spec.energy_num =3;
        optimization_spec.is_signed_svd =0;
     case 'SymDirSigned2D'
       optimization_spec.energy_num =2; 
       optimization_spec.is_signed_svd   =0;
end

switch op.solver
        case 'GD'
            optimization_spec.solver_num =0;
        case 'BCQN'
             optimization_spec.solver_num =1;
end
optimization_spec.is_parallel  = op.parallel_mode;
end