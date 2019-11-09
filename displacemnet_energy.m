%% Copyright @2019. All rights reserved.
%  Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)
%% Computation of weighted simplex-based displacement norm
function delta_energy = displacemnet_energy(cmesh,fV1,fV0,kernelBasis,op)
%op.null = 0;
d =size(cmesh.T,2);
m = size(fV1,2);
Delta = fV1-fV0; 
bounding_rad = max(max(fV0)-min(fV0)); 

if d ==3
    k =3; 

    if op.project_kernel 
        dir = vec2mat(Delta,1);
        kernelProj = kernelBasis*repmat(dir,1,k)*kernelBasis; 
        dir =  dir -sum(kernelProj)';
        Delta = vec2mat(dir,m); 
    end
    DeltaCenterT = (Delta(cmesh.T(:,1)) + Delta(cmesh.T(:,2)) + Delta(cmesh.T(:,3)))/3;
    delta_energy =sum(norms(DeltaCenterT,[],2).*cmesh.Wc); 
  
    mean_edge_length = bounding_rad/ sqrt(cmesh.tn/2);
    delta_energy= delta_energy/mean_edge_length; 
elseif d ==4
    k =size(kernelBasis,1); 
    if op.project_kernel 
        dir = vec2mat(Delta,1);
        kernelProj = kernelBasis*repmat(dir,1,k)*kernelBasis; 
        dir =  dir -sum(kernelProj)';
        Delta = vec2mat(dir,m); 
    end
    DeltaCenterT = ( Delta(cmesh.T(:,1)) + Delta(cmesh.T(:,2))+ Delta(cmesh.T(:,3))+Delta(cmesh.T(:,4)) )/4;
    delta_energy =sum(norms(DeltaCenterT,[],2).*cmesh.Wc); 
    mean_edge_length = bounding_rad/ nthroot(cmesh.tn/6,3); 
    delta_energy= delta_energy/mean_edge_length; 
else
    error('Wrong number of columns in mesh struct');
end
if isfield(op, 'normalize_weight') && op.normalize_weight
    delta_energy = delta_energy/sum(cmesh.Wc);
end

end

