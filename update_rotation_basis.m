%% Copyright @2019. All rights reserved.
%  Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)
function kernelBasis= update_rotation_basis(cmesh,fV,kernelBasis)
    d =size(cmesh.T,2);
    m = size(fV,2);
    
    if d ==3
        if (m~=2)
            error('For a triangle mesh fV should have 2 columns');
        end
        k = size(kernelBasis,1);
        if (k<3)
            error('For a triangle mesh there should 3 vectors in kernelBasis');
        end
        kernelBasis(k,:) = normalize_rows( vec2mat( [-fV(:,2) fV(:,1)],1)');
    elseif d ==4
        if (m~=3)
            error('For a triangle mesh, fV should have 3 columns');
        end
        k = size(kernelBasis,1);
        if (k<6)
            error('For a triangle mesh there should 6 vectors in kernelBasis');
        end
        kernelBasis(4:6,:) = normalize_rows(...
               [ vec2mat( [zeros(cmesh.vn,1) -fV(:,3) fV(:,2)],1)';  ...%rotate around X axis (on YZ plane)
                vec2mat( [-fV(:,3) zeros(cmesh.vn,1) fV(:,1)],1)';  ... %rotate around Y axis (on XZ plane)
                vec2mat( [-fV(:,2) fV(:,1) zeros(cmesh.vn,1)],1)' ] );  %rotate around Z axis (on XY plane)
    else
        error('Wrong number of columns in "T" field of the mesh struct');
    end
end

