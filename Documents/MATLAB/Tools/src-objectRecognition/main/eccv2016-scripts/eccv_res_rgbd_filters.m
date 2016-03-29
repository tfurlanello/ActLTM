
resSaveDir = '/lab/jiaping/papers/ECCV2016/results';


%% (1) comparison between warmstart and scratch

net_alexnet_scratch = '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/fromscratch/exp1/rep-3/iLab20M-iLab_arc_de_dagnn_2streams_alexnet/net-epoch-1.mat';
net_alexnet_deCNN   = '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-iLab20M-deCNN/exp1/rep-3/iLab20M-iLab_arc_de_dagnn_2streams_alexnet/net-epoch-10.mat';

% load alexnet on pretrained deCNN
load(net_alexnet_deCNN);
conv1f = net.params(1).value;
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
% filters_deCNN = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
 filters_deCNN = plotCollage(D, [size(conv1f, 1) size(conv1f,2)], [16 6]);
 
figure; imagesc(filters_deCNN); title('alexnetfromdeCNN');
set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1]);
set(gca, 'ytick', [], 'yticklabel', {}, 'xtick', [], 'xticklabel', {});
set(gcf, 'position', [200 200 1000 1000]);
axis equal; axis tight;

% export_fig(fullfile(resSaveDir, 'vis-rgbd-alexNet-from-deCNN.pdf'), '-pdf', '-m1', '-transparent', gcf);


% load alexNet from scratch
load(net_alexnet_scratch);
conv1f = net.params(1).value;
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
filters_alexnet = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
filters_alexnet = plotCollage(D, [size(conv1f, 1) size(conv1f,2)], [16 6]);


figure; imagesc(filters_alexnet); %title('alexNetfromscratch');
set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1]);
set(gca, 'ytick', [], 'yticklabel', {}, 'xtick', [], 'xticklabel', {});
set(gcf, 'position', [200 200 1000 1000]);
axis equal; axis tight;
% export_fig(fullfile(resSaveDir, 'vis-rgbd-alexnet-from-scratch.pdf'), '-pdf', '-m1', '-transparent', gcf);


%% (2) comparison between deCNN and alexnet
net_alexnet_from_alexnet = '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-iLab20M-AlexNet/exp1/rep-4/iLab20M-iLab_arc_de_dagnn_2streams_alexnet/net-epoch-10.mat';
net_deCNN_from_alexnet   = '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-iLab20M-AlexNet/exp1/rep-4/iLab20M-iLab_arc_de_dagnn_2streams_woL2-w1.000-w1.000/net-epoch-10.mat';

% load alexnet from alexnet
load(net_alexnet_from_alexnet);
conv1f = net.params(1).value;
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
filters_deCNN = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
filters_deCNN = plotCollage(D, [size(conv1f, 1) size(conv1f,2)], [16 6]);


figure; imagesc(filters_deCNN); title('alexnetFromAlexnet');
set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1]);
set(gca, 'ytick', [], 'yticklabel', {}, 'xtick', [], 'xticklabel', {});
set(gcf, 'position', [200 200 1000 1000]);
axis equal; axis tight;

% export_fig(fullfile(resSaveDir, 'vis-rgbd-alexNet-from-alexnet.pdf'), '-pdf', '-m1', '-transparent', gcf);


% load deCNN from alexnet
load(net_deCNN_from_alexnet);
conv1f = net.params(1).value;
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
filters_alexnet = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
filters_alexnet = plotCollage(D, [size(conv1f, 1) size(conv1f,2)], [16 6]);

figure; imagesc(filters_alexnet); %title('deCNNfromAlexNet');
set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1]);
set(gca, 'ytick', [], 'yticklabel', {}, 'xtick', [], 'xticklabel', {});
set(gcf, 'position', [200 200 1000 1000]);
axis equal; axis tight;
% export_fig(fullfile(resSaveDir, 'vis-rgbd-deCNN-from-alexnet.pdf'), '-pdf', '-m1', '-transparent', gcf);


