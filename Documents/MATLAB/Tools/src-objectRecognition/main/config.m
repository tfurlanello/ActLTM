global workdir;
t = pwd;
idx = strfind(t, filesep);
workdir = t(1:(idx(end)-1)); 
addpath(genpath(fullfile(workdir, 'matconvnet-1.0-beta14')));
% addpath(genpath(fullfile(workDir, 'vlfeat-0.9.20')));
addpath(genpath(fullfile(workdir, 'main')));
addpath(genpath(fullfile(workdir, 'Felzenszwalb-Segmentation-matlab')));
addpath(genpath(fullfile(workdir, 'PolygonClipper')));
addpath(genpath(fullfile(workdir, 'CaptureFigVid')));
% run(fullfile(workDir, 'vlfeat-0.9.20', 'toolbox', 'vl_setup.m'));
 
setupScript = fullfile(workdir, 'matconvnet-1.0-beta14', 'matlab', 'vl_setupnn.m');
eval(['run' ' ' setupScript]);

% setupSeg = fullfile(workDir, 'Felzenszwalb-Segmentation-matlab', 'compileFelzenszwalbSegmentation.m');
% eval(['run' ' ' setupSeg]);
   
 
