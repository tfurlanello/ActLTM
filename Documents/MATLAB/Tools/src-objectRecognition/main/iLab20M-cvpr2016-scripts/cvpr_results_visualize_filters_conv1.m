% get the first layer filters

saveDir = '/lab/jiaping/projects/iLab-object-recognition/results/CVPR2016';


alexNetFile       =  '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-obj/net-epoch-15.mat';
w2cnn_I_NetFile   =  '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-fc2/net-epoch-15.mat';
w2cnn_MI_NetFile  =  '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/net-epoch-15.mat';



% alexnet
load(alexNetFile);
conv1f = net.params(1).value;
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
multiLevelObjEnv = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
% figure; 
% imshow(multiLevelObjEnv);
set(gcf, 'Units', 'normalized', 'Position', [0,0,1,1] );
export_fig(fullfile(saveDir,'conv1-alexnet.png'), '-png', '-m1',   gcf );
export_fig(fullfile(saveDir, 'conv1-alexnet.pdf'), '-pdf', '-m1',   gcf );
close all;

% 2w-cnn-I
load(w2cnn_I_NetFile);
conv1f = net.params(1).value;
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
multiLevelObjEnv = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
% figure; 
% imshow(multiLevelObjEnv);
set(gcf, 'Units', 'normalized', 'Position', [0,0,1,1] );
export_fig(fullfile(saveDir,'conv1-2W-CNN-I.png'), '-png', '-m1',   gcf );
export_fig(fullfile(saveDir, 'conv1-2W-CNN-I.pdf'), '-pdf', '-m1',   gcf );
close all;

% 2w-cnn-mi
load(w2cnn_MI_NetFile);
conv1f = net.params(1).value;
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
multiLevelObjEnv = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
% figure; 
% imshow(multiLevelObjEnv);
set(gcf, 'Units', 'normalized', 'Position', [0,0,1,1] );
export_fig(fullfile(saveDir,'conv1-2W-CNN-MI.png'), '-png', '-m1',   gcf );
export_fig(fullfile(saveDir, 'conv1-2W-CNN-MI.pdf'), '-pdf', '-m1',   gcf );
close all;
