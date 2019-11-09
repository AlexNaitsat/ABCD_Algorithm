%% Copyright @2019. All rights reserved.
%  Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

%% wrapper to run mex optimization of a single measure 
function [mesh,results,dir] = ABCD_single_measure_wrapper(cmesh,mesh,meshData,op,distS)

run_mex =1;  
mesh.bbox      = [max(mesh.fV); min(mesh.fV)];
mesh.diameter  = max(mesh.bbox(1,:) - mesh.bbox(2,:)); 

%result reporting 
results.dist = zeros(1,op.iter_per_cycle+1);
results.degen_num = zeros(1,op.iter_per_cycle+1);
results.flip_num  = zeros(1,op.iter_per_cycle+1);
results.block_num  = zeros(1,op.iter_per_cycle);
op.skip_fixed_vertices =1; 

fV0=mesh.fV; %save intial target vetices for displacement  norm
opLS = OptimizationParams.SubStruct(op,'LS');
opBlock = OptimizationParams.SubStruct(op,'block');

for outer_i=1:op.iter_per_cycle
    opLS.penalize_flips_t =logical(0); 
    optimizer_spec = GetMexedSolverSpec(op);
    optimizer_spec.source_dim = mesh.source_dim;

    optimizer_spec.K_hat = opBlock.block_threshold;
    optimizer_spec.single_fixed_block=0;
    
    [fV_new, grad,~,SV_t,~,E,runtimes] = ABCD_mex_solver(...
                    cmesh.V(:,1:mesh.source_dim), mesh.fV(:,1:2), cmesh.T,...
                    optimizer_spec,mesh.fixed_elements,meshData);
    
    mesh.fV(:,1:2) = fV_new;
    
    distS.grad = grad;
    distS.SV= SV_t;
    distS.E=E;
    results.dist(outer_i+1)= E/cmesh.WcSum;
    fliped_num = sum(distS.SV(:,1)<0 | distS.SV(:,2)<0);
    degen_num  = sum( abs(distS.SV(:,2))<op.sing_eps);
    mesh.kernelBasis= update_rotation_basis(cmesh,fV0,mesh.kernelBasis);
    results.delta_energy = displacemnet_energy(cmesh,mesh.fV,fV0,mesh.kernelBasis,op);
    results.distS=distS;
    
    disp([ op.dist_name '=' num2str(results.dist(outer_i+1)) ' ,' num2str(fliped_num) ...
        '-flipped, ' num2str(degen_num) '-degenerate simplices, ', num2str(results.delta_energy) '-displacement']);
    results.flip_num(outer_i+1)= fliped_num;
    results.degen_num(outer_i+1)= degen_num;
    results.mex_runtime(outer_i+1) =  sum(runtimes(3:5));
end
end

