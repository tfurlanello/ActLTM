% results analysis:
% 1. label layer: object category
% 2. label layer: camera parameter
% 3. label layer: object + camera


%% read test data and labels
expDir = '/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera';

%% (1) label layer: object category
imdb    =   load(fullfile(expDir, 'iLab20M-alexnet-simplenn-cat2', 'imdb.mat'));
imdb.imageDir = expDir;
netCat  =   load(fullfile(expDir, 'iLab20M-alexnet-simplenn-cat2', 'net-epoch-20.mat'));
netCat  =   netCat.net;

opts_batch = netCat.normalization ;
opts_batch.numThreads  = 12;
opts_batch.transformation = 'stretch' ;
opts_batch.object = true;

getBatch = iLab_getBatchSimpleNNWrapper(opts_batch);
subset =  find(imdb.images.set==3);

opts_fetch.batchSize = 128 ;
opts_fetch.numSubBatches = 1 ;
opts_fetch.gpus = 1 ;
opts_fetch.prefetch = false ;
opts_fetch.sync = false ;
opts_fetch.cudnn = true ;

% [labels_gt_cat_simplenn, labels_pred_cat_simplenn] = ...
%     iLab_predict_simplennBatch(netCat, imdb, subset, getBatch, opts_fetch);

%% (1) label layer: camera-rotation 
imdb    =   load(fullfile(expDir, 'iLab20M-alexnet-simplenn-cam', 'imdb.mat'));
imdb.imageDir = expDir;
netCam  =   load(fullfile(expDir, 'iLab20M-alexnet-simplenn-cam', 'net-epoch-20.mat'));
netCam  =   netCam.net;

opts_batch = netCam.normalization ;
opts_batch.numThreads  = 12;
opts_batch.transformation = 'stretch' ;
opts_batch.object = false;

getBatch = iLab_getBatchSimpleNNWrapper(opts_batch);
subset =  find(imdb.images.set==3);

opts_fetch.batchSize = 128 ;
opts_fetch.numSubBatches = 1 ;
opts_fetch.gpus = 1 ;
opts_fetch.prefetch = false ;
opts_fetch.sync = false ;
opts_fetch.cudnn = true ;

[labels_gt_cam_simplenn, labels_pred_cam_simplenn] = ...
    iLab_simplenn_predictBatch(netCam, imdb, subset, getBatch, opts_fetch);


%% (3) label layer: object category & camera parameters
import dagnn.*;
imdb    =   load(fullfile(expDir, 'iLab20M-alexnet-dagnn-catcam', 'imdb.mat'));
imdb.imageDir = expDir;
netCatCam  =   load(fullfile(expDir, 'iLab20M-alexnet-dagnn-catcam', 'net-epoch-10.mat'));
netCatCam_  =   netCatCam.net;

netCatCam = dagnn.DagNN.loadobj(netCatCam_) ;
netCatCam.accumulateParamDers = 1;
netCatCam.conserveMemory = 0;

opts_batch = netCatCam.meta.normalization ;
opts_batch.numThreads  = 12;
opts_batch.useGpu = true;
opts_batch.transformation = 'stretch' ;

getBatch = iLab_getBatchDagNNWrapper(opts_batch);
subset =  find(imdb.images.set==3);

opts_fetch.batchSize = 128 ;
opts_fetch.numSubBatches = 1 ;
opts_fetch.gpus = 1 ;
opts_fetch.prefetch = false ;
opts_fetch.sync = false ;
opts_fetch.cudnn = true ;

[labels_gt_cat_dagnn, labels_pred_cat_dagnn, labels_gt_cam_dagnn, labels_pred_cam_dagnn]  = ...
    iLab_dagnn_predictBatch(netCatCam, imdb, subset, getBatch, opts_fetch);


%% (4) label layer: object category & camera parameters
import dagnn.*;
imdb    =   load(fullfile(expDir, 'iLab20M-alexnet-dagnn-catcam-structured', 'imdb.mat'));
imdb.imageDir = expDir;
netCatCam  =   load(fullfile(expDir, 'iLab20M-alexnet-dagnn-catcam-structured', 'net-epoch-10.mat'));
netCatCam_  =   netCatCam.net;

netCatCam = dagnn.DagNN.loadobj(netCatCam_) ;
netCatCam.accumulateParamDers = 1;
netCatCam.conserveMemory = 0;

opts_batch = netCatCam.meta.normalization ;
opts_batch.numThreads  = 12;
opts_batch.useGpu = true;
opts_batch.transformation = 'stretch' ;

getBatch = iLab_getBatchDagNNWrapper(opts_batch);
subset =  find(imdb.images.set==3);

opts_fetch.batchSize = 128 ;
opts_fetch.numSubBatches = 1 ;
opts_fetch.gpus = 1 ;
opts_fetch.prefetch = false ;
opts_fetch.sync = false ;
opts_fetch.cudnn = true ;

[labels_gt_cat_dagnn_s, labels_pred_cat_dagnn_s, labels_gt_cam_dagnn_s, labels_pred_cam_dagnn_s]  = ...
    iLab_dagnn_predictBatch(netCatCam, imdb, subset, getBatch, opts_fetch);


save('/lab/igpu3/iLab-cnn-res-structured.mat', 'labels_gt_cat_simplenn', 'labels_pred_cat_simplenn', ...
                 'labels_gt_cam_simplenn', 'labels_pred_cam_simplenn', ...
                 'labels_gt_cat_dagnn', 'labels_pred_cat_dagnn', 'labels_gt_cam_dagnn', 'labels_pred_cam_dagnn' ,...
                 'labels_gt_cat_dagnn_s', 'labels_pred_cat_dagnn_s', 'labels_gt_cam_dagnn_s', 'labels_pred_cam_dagnn_s');


