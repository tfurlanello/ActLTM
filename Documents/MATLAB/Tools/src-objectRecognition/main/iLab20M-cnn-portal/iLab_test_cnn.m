% run the trained model on the test data

%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                   parameters to get batch
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
simplenn_opts_getbatch.numThreads       =   12;
simplenn_opts_getbatch.transformation   =   'stretch' ;
simplenn_opts_getbatch.numAugments      =   1;
simplenn_opts_getbatch = iLab_nn_validateGetImageBatchParam(simplenn_opts_getbatch);

dagnn_opts_getbatch.numThreads      =   12;
dagnn_opts_getbatch.useGpu          =   true;
dagnn_opts_getbatch.transformation  =   'stretch' ;
dagnn_opts_getbatch.numAugments     =   1;
dagnn_opts_getbatch = iLab_nn_validateGetImageBatchParam(dagnn_opts_getbatch);


%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                    parameters to run test
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
nn_opts_runTest.batchSize       = 128 ;
nn_opts_runTest.numSubBatches   = 1 ;
nn_opts_runTest.prefetch        = false ;
nn_opts_runTest.gpus            = 1 ;
nn_opts_runTest.sync            = false ;
nn_opts_runTest.cudnn           = true ;
nn_opts_runTest.conserveMemory  = false ;
nn_opts_runTest = iLab_nn_validateRunTestParam(nn_opts_runTest);


%==========================================================================
%                                                         working directory
%==========================================================================
expDir = '/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera';


%==========================================================================
%%                            experiment 1:  object prediction only
%==========================================================================
imdb    =   load(fullfile(expDir, 'iLab20M-alexnet-simplenn-cat2', 'imdb.mat'));
imdb.imageDir = expDir;
netObj  =   load(fullfile(expDir, 'iLab20M-alexnet-simplenn-cat2', 'net-epoch-20.mat'));
netObj  =   netObj.net;

opts_getbatch   = vl_argparse(simplenn_opts_getbatch, netObj.normalization) ;
getBatch        = iLab_getBatchSimpleNNWrapper(opts_getbatch, {'input', 'label_obj'});
subset          =  find(imdb.images.set==3);

[labels_gt_obj_simplenn, labels_pred_obj_simplenn] = ...
    iLab_simplenn_predictBatch(netObj, imdb, subset, getBatch, nn_opts_runTest);




%==========================================================================
%%                            experiment 2:  object prediction only
%==========================================================================
imdb    =   load(fullfile(expDir, 'iLab20M-alexnet-simplenn-cam', 'imdb.mat'));
imdb.imageDir = expDir;
netEnv  =   load(fullfile(expDir, 'iLab20M-alexnet-simplenn-cam', 'net-epoch-20.mat'));
netEnv  =   netEnv.net;

opts_getbatch   = vl_argparse(simplenn_opts_getbatch, netEnv.normalization) ;
getBatch        = iLab_getBatchSimpleNNWrapper(opts_getbatch, {'input', 'label_env'});
subset          =  find(imdb.images.set==3);

[labels_gt_env_simplenn, labels_pred_env_simplenn] = ...
    iLab_simplenn_predictBatch(netEnv, imdb, subset, getBatch, nn_opts_runTest);




%==========================================================================
%%     experiment 3:  object and environment prediction (unstructured)
%==========================================================================
import dagnn.*;
imdb            =   load(fullfile(expDir, 'iLab20M-alexnet-dagnn-catcam', 'imdb.mat'));
imdb.imageDir   =   expDir;
netObjEnv       =   load(fullfile(expDir, 'iLab20M-alexnet-dagnn-catcam', 'net-epoch-10.mat'));
netObjEnv_      =   netObjEnv.net;

netObjEnv = dagnn.DagNN.loadobj(netObjEnv_) ;
netObjEnv.accumulateParamDers = 1;
netObjEnv.conserveMemory = 0;

opts_getbatch   = vl_argparse(dagnn_opts_getbatch, netObjEnv.meta.normalization);
getBatch        = iLab_getBatchDagNNWrapper(opts_getbatch, {'input', 'label_obj', 'label_env'});
subset          =  find(imdb.images.set==3);

[labels_gt_obj_dagnn, labels_pred_obj_dagnn, labels_gt_env_dagnn, labels_pred_env_dagnn]  = ...
    iLab_dagnn_predictBatch(netObjEnv, imdb, subset, getBatch, ...
                                {'prediction_obj', 'prediction_env'}, nn_opts_runTest);


%==========================================================================
%%     experiment 4:  object and environment prediction (structured)
%==========================================================================
import dagnn.*;
imdb    =   load(fullfile(expDir, 'iLab20M-alexnet-dagnn-catcam-structured', 'imdb.mat'));
imdb.imageDir = expDir;
netObjEnv  =   load(fullfile(expDir, 'iLab20M-alexnet-dagnn-catcam-structured', 'net-epoch-10.mat'));
netObjEnv_  =   netObjEnv.net;

netObjEnv = dagnn.DagNN.loadobj(netObjEnv_) ;
netObjEnv.accumulateParamDers = 1;
netObjEnv.conserveMemory = 0;

opts_getbatch   = vl_argparse(dagnn_opts_getbatch, netObjEnv.meta.normalization);
getBatch        = iLab_getBatchDagNNWrapper(opts_getbatch, {'input', 'label_obj', 'label_env'});
subset          =  find(imdb.images.set==3);

[labels_gt_obj_dagnn_s, labels_pred_obj_dagnn_s, labels_gt_env_dagnn_s, labels_pred_env_dagnn_s]  = ...
    iLab_dagnn_predictBatch(netObjEnv, imdb, subset, getBatch, ...
                                {'prediction_obj', 'prediction_env'}, nn_opts_runTest);

 
