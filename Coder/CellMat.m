%% Copyright @2019. All rights reserved.
%  Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

%%Class that implment nx1 cell array and compatible with matlab coder.
classdef CellMat
    properties
        mat;
        pos;
        n;
        d;
    end
    
    methods
        function obj=CellMat(cell_arr,d)
            obj.d=d;
            obj.mat = cell2mat(cell_arr);
            dims = cellfun(@(M) size(M,d),cell_arr);
            [m,n]=size(dims);
            if (m > n)
                dims = dims';
                n=m;
            end
            obj.pos = zeros(n,2);
            obj.pos(:,1) = cumsum([1 dims(1:end-1)]);
            obj.pos(:,2) = cumsum(dims);
            
            obj.n = n;
        end
        
        function val = at(obj,i)
            if obj.d==1
                val = obj.mat( obj.pos(i,1): obj.pos(i,2),:);
            else
                val = obj.mat(:, obj.pos(i,1): obj.pos(i,2));
            end
        end
    end
end

