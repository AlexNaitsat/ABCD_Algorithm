function visualize_anchor_2D(anchor,anchor_opt)
anchor_options.segment_num =20;
t=linspace(0,2*pi,anchor_options.segment_num);
for p = anchor'
    x = repmat(p(1),1,anchor_opt.segment_num)+anchor_opt.rad*cos(t);
    y = repmat(p(2),1,anchor_opt.segment_num)+anchor_opt.rad*sin(t);
    patch(x, y,anchor_opt.spec{:});
end

