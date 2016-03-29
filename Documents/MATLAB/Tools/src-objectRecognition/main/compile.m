global workdir;
t = pwd;
idx = strfind(t, filesep);
workdir = t(1:(idx(end)-1)); 

addpath(genpath(fullfile(workdir, 'Felzenszwalb-Segmentation-matlab')));
setupSeg = fullfile(workdir, 'Felzenszwalb-Segmentation-matlab', 'compileFelzenszwalbSegmentation.m');
eval(['run' ' ' setupSeg]);