%2D/surface mesh visualization function
function fig=draw_opt_results_Surf_analytic(mesh,fV0,dist_name,dis,options )

options.null = 0;
if ~isfield(options,'draw_mesh');    options.draw_mesh=0;     end
if ~isfield(options,'draw_colbar');  options.draw_colbar=0;   end
if ~isfield(options, 'mark_fixed_vertices'); options.mark_fixed_vertices=0; end
if ~isfield(options,'draw_trimesh'); options.draw_trimesh=0; end %if set dont fix boundary.
if ~isfield(options,'tet_list'); options.tet_list=[]; end;

m=size(fV0,2);

if isempty(fV0)
    fV0=mesh.fV;
end
if  options.color_range(1)  > options.color_range(2)
    if (options.color_range(2) == -inf)
        options.color_range(2) = inf;
    else
        options.color_range(2) =options.color_range(1)+10^-3*options.color_range(1);
    end
end

dis_field = 'dis_v';
dis.v = (dis.v)./mesh.RingWcSum; 

if options.log_dist_scale %colors in a logarithmic scale
    dis.v = log(abs(dis.v));
    dis.t = log(abs(dis.t));
end
mesh.dis_v = dis.v;
mesh.dis_t = dis.t;
mesh.SV_t = dis.SV;
if options.check_flips
    mesh.overpaint_tri= mesh.T(mesh.SV_t(:,1) <=0,:);
    options.FaceColor = options.flipped_cells_color;
end
d_opt = options;
d_opt.dist = [dis_field(1:end-1) 't']; 

fig = gcf;
fig.Color = options.bg_color;
    %% drawing stage
    if m==2 
        fV0 = [fV0 zeros(mesh.vn,1)];
        mesh.fV = [mesh.fV zeros(mesh.vn,1)];
    end
    draw_3D_triangles(options.tet_list,mesh,fV0,d_opt);
    if isfield(options,'view')
        view(options.view);
    end
    
    if isfield(options,'color_range')
        caxis manual; caxis([options.color_range(1) options.color_range(2)]);
    end
    colormap(options.col_map);

    if ~options.show_axis;  axis off; end
    hold on
end