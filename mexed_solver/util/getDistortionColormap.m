function [cols] = getDistortionColormap()
%GETDISTORTIONCOLORMAP Summary of this function goes here
%   Detailed explanation goes here
CBcols = [0.9 0.9 0.9];
t=(1:64) / 64; t = t';
cols = (1-t)*CBcols + t*[0.7 0 0];
cols(cols>1) = 1;
caxis([-5 10]);
end

