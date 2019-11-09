%% Copyright @2019. All rights reserved.
%  Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

%% Visualization function for triangle meshes
function  draw_3D_triangles(lT,tet,V,options)
options.null =0;
if isempty(lT); lT=1:tet.tn; end;
if ~exist('V','var') || isempty(V)
    V = tet.V;
end

dist_name  = [options.dist([1:end-1]) 'v']; 
dist=tet.(dist_name);
h=trimesh(tet.T(lT,:),V(:,1),V(:,2),V(:,3),'FaceVertexCData',dist,'FaceColor','interp','Facealpha',options.alpha,'EdgeColor',options.EdgeColor,'LineWidth',options.line_width,'EdgeAlpha',options.EdgeAlpha);

if isfield(tet,'overpaint_tri') && isfield(options,'FaceColor') 
    hold on
    h_over=trimesh(tet.overpaint_tri,V(:,1),V(:,2),V(:,3),'FaceColor',options.FaceColor,...
          'Facealpha',options.alpha,'EdgeColor',options.EdgeColor,'EdgeAlpha',options.EdgeAlpha);
end

if isfield(options,'use_shading') && options.use_shading
    if isfield(options,'shad')
        shad_fields = fieldnames(options.shad);
        for i=1:length(shad_fields)
            h.(shad_fields{i}) = options.shad.(shad_fields{i});
            h_over.(shad_fields{i}) = options.shad.(shad_fields{i});
        end
    end
    
    if isfield(options,'shad1')
        shad_fields = fieldnames(options.shad1);
        for i=1:length(shad_fields)
            h1.(shad_fields{i}) = options.shad1.(shad_fields{i});
        end
    end
    %setting light according to 'lh' field;
    if isfield(options,'lh')
        for i=1:length(options.lh);  light(options.lh(i)); end
    end
end

if options.axis_equal; axis equal; end

if isfield(options,'mark_fixed_vertices') && options.mark_fixed_vertices
    hold on
    fixed_V=V(logical(tet.is_fixed_v),:);
    visualize_anchor_2D(fixed_V,options.anchor_opt);
end
end

