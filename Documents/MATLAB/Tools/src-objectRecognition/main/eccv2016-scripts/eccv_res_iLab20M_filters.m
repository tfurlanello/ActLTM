
resSaveDir = '/lab/jiaping/papers/ECCV2016/results';


resDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2';
deCNN_folder    =  'iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000';
alexNet_folder  =  'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';


net_alexnet = fullfile(resDir, 'f56', alexNet_folder, 'net-epoch-6.mat');
net_deCNN   = fullfile(resDir, 'f56', deCNN_folder, 'net-epoch-6.mat');


% net_alexnet = '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016/pretrained-iLab20M-nobnorm/iLab20M-iLab_arc_de_dagnn_2streams_alexnet/net-epoch-6.mat';
% net_deCNN = '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016/pretrained-iLab20M-nobnorm/iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000/net-epoch-6.mat';


% load deCNN
load(net_deCNN);
conv1f = net.params(1).value;
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
filters_deCNN = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
filters_deCNN = plotCollage(D, [size(conv1f, 1) size(conv1f,2)], [16 6]);

 figure; imagesc(filters_deCNN);% title('deCNN');
set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1]);
set(gca, 'ytick', [], 'yticklabel', {}, 'xtick', [], 'xticklabel', {});
set(gcf, 'position', [200 200 1000 1000]);
axis equal; axis tight;

% export_fig(fullfile(resSaveDir, 'vis-iLab20M-disCNN.pdf'), '-pdf', '-m1', '-transparent', gcf);


% load alexNet
load(net_alexnet);
conv1f = net.params(1).value;
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
filters_alexnet = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
 filters_alexnet = plotCollage(D, [size(conv1f, 1) size(conv1f,2)], [16 6]);

figure; imagesc(filters_alexnet); %title('alexNet');
set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1]);
set(gca, 'ytick', [], 'yticklabel', {}, 'xtick', [], 'xticklabel', {});
set(gcf, 'position', [200 200 1000 1000]);
axis equal; axis tight;
% export_fig(fullfile(resSaveDir, 'vis-iLab20M-alexnet.pdf'), '-pdf', '-m1', '-transparent', gcf);
