
resDir =  '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal';


iLab_alexNet_2imageNet_res      =   load(fullfile(resDir, 'imageNet-iLab_alexnet.mat'));
iLab_2WCNN_2imageNet_res        =   load(fullfile(resDir, 'imageNet-iLab_2WCNN.mat'));
imageNet_alexNet_2imageNet_res  =   load(fullfile(resDir, 'imageNet-imageNet_alexnet.mat'));


meta2012_file = '/lab/jiaping/igpu3home2/u/jiaping/imageNet2012/ILSVRC2012_devkit_t12/data/meta.mat';
meta2010_file = '/lab/jiaping/igpu3home2/u/jiaping/imageNet2010/devkit-1.0/data/meta.mat';

meta2012 = load(meta2012_file);
meta2010 = load(meta2010_file);


winid2012 = {meta2012.synsets(1:1000).WNID};
words2012 = {meta2012.synsets(1:1000).words};

winid2010 = {meta2010.synsets(1:1000).WNID};
words2010 = {meta2010.synsets(1:1000).words};


% database files
ILSVR2010test_imdb_file = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal/imdb2010test.mat';
ILSVR2012val_imdb_file = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal/imdb2012val.mat';

load(ILSVR2010test_imdb_file);
imdb2010test  = imdb;

load(ILSVR2012val_imdb_file);
imdb2012val = imdb;

% imdb of iLab-20M dataset
iLab20M_imdb_file = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal/iLab20M-alexnet-dagnn-STL-multiLevelInjection-conv1234fc2-1024/imdb.mat';
iLab20M_imdb = load(iLab20M_imdb_file);
iLab20M_classes = iLab20M_imdb.classes.name{1};


% imagenet ground truth labels
imageNet_imdb2010 = load('/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal/imdb2010test.mat');
imageNet_imdb2012 = load('/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal/imdb2012val.mat');


gt2012 = imageNet_imdb2012.imdb.images.gtlabel;
gt2012 = gt2012(1,:);

gt2010 = imageNet_imdb2010.imdb.images.gtlabel;
gt2010 = gt2010(1,:);


%=======================================================================
%%                                                           2012
%=========================================================================


%% prediction accuracies: 2W-CNN
res_2wcnn_2012 = iLab_2WCNN_2imageNet_res.labels_iLab_2WCNN_2012val;
res_2wcnn_2012_pred = res_2wcnn_2012(1).prediction;

%% prediction accuracies: imageNet_alexnet
res_imageNet_alexNet_2012 = imageNet_alexNet_2imageNet_res.labels_imageNet_alexnet_2012val;
res_imageNet_alexNet_2012_pred = res_imageNet_alexNet_2012(1).prediction;

%% prediction accuracies: iLab_alexnet
res_iLab_alexNet_2012 = iLab_alexNet_2imageNet_res.labels_iLab_alexnet_2012val;
res_iLab_alexNet_2012_pred = res_iLab_alexNet_2012(1).prediction;


%% object: pickup
% pickup, 281
% tank: 250
% sports car: 274
% racing car

obj_gt_idx2012 = find(gt2012 == 250);

for i=1:numel(obj_gt_idx2012)
    imname = imageNet_imdb2012.imdb.images.name{obj_gt_idx2012(i)};
    im = imread(fullfile(imageNet_imdb2012.imdb.imageDir, imname));
    imwrite(im, fullfile('imagenet-check', imname));
    
end



obj_pred_2wcnn = res_2wcnn_2012_pred(obj_gt_idx2012);
obj_pred_iLab_alexNet = res_iLab_alexNet_2012_pred(obj_gt_idx2012);
obj_pred_imageNet_alexnet = res_imageNet_alexNet_2012_pred(obj_gt_idx2012);

figure; hist(obj_pred_2wcnn, 1:1:10); 
count_2wcnn = hist(obj_pred_2wcnn, 1:1:10); 
set(gca, 'xtick', 1:10, 'xticklabel', iLab20M_classes);
title('2wcnn');


figure; hist(obj_pred_iLab_alexNet, 1:1:10); 
count_ilab_alexnet =  hist(obj_pred_iLab_alexNet, 1:1:10); 
set(gca, 'xtick', 1:10, 'xticklabel', iLab20M_classes);
title('iLab-alexnet');

figure; hist(obj_pred_imageNet_alexnet); 
 count_imagenet_alexnet = hist(obj_pred_imageNet_alexnet); 




%=======================================================================
%%                                                           2010
%=========================================================================


%% prediction accuracies: 2W-CNN
res_2wcnn_2010 = iLab_2WCNN_2imageNet_res.labels_iLab_2WCNN_2010test;
res_2wcnn_2010_pred = res_2wcnn_2010(1).prediction;

%% prediction accuracies: imageNet_alexnet
res_imageNet_alexNet_2010 = imageNet_alexNet_2imageNet_res.labels_imageNet_alexnet_2010test;
res_imageNet_alexNet_2010_pred = res_imageNet_alexNet_2010(1).prediction;

%% prediction accuracies: iLab_alexnet
res_iLab_alexNet_2010 = iLab_alexNet_2imageNet_res.labels_iLab_alexnet_2010test;
res_iLab_alexNet_2010_pred = res_iLab_alexNet_2010(1).prediction;


%% object: pickup
% pickup, 683
% tank: 663
% plane: 778
% subway train: 795
% racing car 676

obj_gt_idx2010 = find(gt2010 == 663);

for i=1:numel(obj_gt_idx2010)
    imname = imageNet_imdb2010.imdb.images.name{obj_gt_idx2010(i)};
    im = imread(fullfile(imageNet_imdb2010.imdb.imageDir, imname));
    imwrite(im, fullfile('imagenet-check', imname));
    
end



obj_pred_2wcnn_2010 = res_2wcnn_2010_pred(obj_gt_idx2010);
obj_pred_iLab_alexNet_2010 = res_iLab_alexNet_2010_pred(obj_gt_idx2010);
obj_pred_imageNet_alexnet_2010 = res_imageNet_alexNet_2010_pred(obj_gt_idx2010);

figure; hist(obj_pred_2wcnn_2010, 1:1:10); 
count_2wcnn = hist(obj_pred_2wcnn_2010, 1:1:10); 
set(gca, 'xtick', 1:10, 'xticklabel', iLab20M_classes);
title('2wcnn-2010');


figure; hist(obj_pred_iLab_alexNet_2010, 1:1:10); 
count_iLab_alexnet = hist(obj_pred_iLab_alexNet_2010, 1:1:10); 
set(gca, 'xtick', 1:10, 'xticklabel', iLab20M_classes);
title('iLab-alexnet-2010');

figure; hist(obj_pred_imageNet_alexnet_2010); 
count_imageNet_alexNet = hist(obj_pred_imageNet_alexnet_2010); 
 

