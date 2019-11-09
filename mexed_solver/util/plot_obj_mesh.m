function [ ] = plot_obj_mesh( vertex, face, frame,fig )

TR = triangulation(face, vertex);
indB = TR.freeBoundary();
intB = [indB(:, 1); indB(1, 1)];
boundary_line_width = 1;
if ~exist('fig','var'); fig=1; end
file = figure(fig);

clf

min_min = min(vertex);
max_max = max(vertex);
min_x = min_min(1);
min_y = min_min(2);
max_x = max_max(1);
max_y = max_max(2);
cen_x = 0.5 * (min_x + max_x);
cen_y = 0.5 * (min_y + max_y);
radius = max(max_x - min_x, max_y - min_y);
radius = radius * 0.6;

axis([cen_x - radius cen_x + radius cen_y - radius cen_y + radius]);
caxis([0 10])

patch('Faces', face, 'Vertices', vertex, 'FaceVertexCData', ones(size(vertex, 1), 1), 'FaceColor','flat', 'EdgeAlpha',0.1);

colormap(getDistortionColormap())

line(vertex(intB, 1, end), vertex(intB, 2, end), 'color', 'b', 'linewidth', boundary_line_width);

axis square
% axis off
title(frame);
drawnow;

% getframe;
% set(gca,'units','centimeters')
% set(gca,'Position',[1.8 0.8 14.8 14.8]);
% set(gcf, 'PaperUnits','centimeters');
% set(gcf, 'PaperSize', [12 12]);
% set(gcf, 'PaperPositionMode', 'manual');
% set(gcf, 'PaperPosition',[0 0 12 12]);
% print(file, '-dbmp', sprintf(strcat('../result/image%d.bmp'), frame), '-r400')  

end

