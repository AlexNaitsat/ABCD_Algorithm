%% Copyright @2019. All rights reserved.
%  Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

%% Class that implment nx1 cell array and compatible with matlab coder.
classdef CellVec
    properties
        vec;
        pos;
        n;
    end
    
    methods
        function obj=CellVec(cell_arr)
            [m,n]=size(cell_arr);
            if iscell(cell_arr)
                s1  =  max(cellfun(@(V) size(V,1),cell_arr));
                s2  =  max(cellfun(@(V) size(V,2),cell_arr));
                if  s1 > m || s2 > n 
                    obj.vec = cell2mat(cell_arr');
                else
                    obj.vec = cell2mat(cell_arr);
                end
                dims = cellfun(@length,cell_arr);
                [m,n]=size(dims);
                if (m > n)
                    dims = dims';
                    n=m;
                end
            else
               [n,m]=size(cell_arr);
               mat = cell_arr';
               obj.vec = mat(:)';
               dims = repmat(m,1,n);
            end
            
            obj.pos = zeros(n,2);
            obj.pos(:,1) = cumsum([1 dims(1:end-1)]);
            obj.pos(:,2) = cumsum(dims);
            
            obj.n = n;
        end
        
        
        function val = at(obj,i)
            val = obj.vec( obj.pos(i,1): obj.pos(i,2) );
        end
    end
end

