function dist_min = dist_minimum(dist_name,d)

dist_min = 0;

switch  dist_name
    case 'ARAP'
        dist_min =0;
    case 'K_2D'
        dist_min =1;
    case 'Ksigned_2D'
        dist_min =1;
    case 'Csigned_2D'
        dist_min =1;
    case 'C_2D'
        dist_min =1;
    case 'SymDir2D'  %Symmetric Dirichlet
        dist_min =4;
        
    case 'SymDirSigned2D'  %Symmetric Dirichlet
        dist_min =4;
    case 'SymDir3D'  %Symmetric Dirichlet
          dist_min=6;
    case 'SymDirSigned3D' 
          dist_min=6;
    case 'SymmDirUnified'
        if (d==4) 
            dist_min=6;
        else
            dist_min=4;
        end
    
    case 'V_2D'
        dist_min =1;
    case 'KplusC_2D'
        dist_min =1;
    otherwise
        error(['No such distorion measure:' dist_name ]);
end
