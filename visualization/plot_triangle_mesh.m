%drawing triangle mesh with color coded distortions 
function  plot_triangle_mesh(T,V,element_dist, sing_values, options,is_fixed_v)
options.null =0;
if ~isfield(options,'FaceAlpha')  ; options.FaceAlpha= 1; end
if ~isfield(options,'EdgeAlpha')  ; options.EdgeAlpha= 1; end
if ~isfield(options,'EdgeColor')   ; options.EdgeColor   =  [0 0 0]; end
if ~isfield(options,'FaceColor') ;    options.FaceColor =  [1 1 0]; end
if ~isfield(options,'axis_equal'); options.axis_equal =1; end

if ~isfield(options,'line_width'); options.line_width =0.5; end
if ~isfield(options,'col_map');   options.col_map =cool(50);end

vn = size(V,1);
if size(V,2)==2
    V =  [V zeros(vn,1)];
end

trimesh(T,V(:,1),V(:,2),V(:,3),element_dist, 'FaceColor','flat','Facealpha',options.FaceAlpha,'EdgeColor',...
    options.EdgeColor,'LineWidth', options.line_width,'EdgeAlpha', options.EdgeAlpha  );


hold on
is_flliped  = (sing_values(:,1).*sing_values(:,2))<0;
trimesh(T(is_flliped,:),V(:,1),V(:,2),V(:,3),'FaceColor','y','FaceAlpha',options.FaceAlpha,...
    'EdgeColor', options.EdgeColor,'LineWidth', options.line_width,'EdgeAlpha', options.EdgeAlpha);

if options.axis_equal; axis equal; end

if isfield(options,'color_range')
    caxis manual;
    if options.color_range(1) < options.color_range(2)
        caxis([options.color_range(1) options.color_range(2)]);
    else
        caxis([options.color_range(1) options.color_range(1)+1]);
    end
end
colormap(options.col_map);

if isfield(options,'mark_fixed_vertices') && options.mark_fixed_vertices
    hold on
    fixed_V=V(logical(is_fixed_v),:);
    scatter3(fixed_V(:,1),fixed_V(:,2),fixed_V(:,3),100,'blue','filled','MarkerEdgeColor','black');
end

if isfield(options,'view')
    view(options.view);
end
end