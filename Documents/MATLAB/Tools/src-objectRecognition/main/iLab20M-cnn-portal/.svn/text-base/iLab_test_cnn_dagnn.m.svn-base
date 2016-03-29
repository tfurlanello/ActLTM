% run trained dagnn model on test data

%==========================================================================
%                                                         working directory
%==========================================================================
expDir = '/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera';

import dagnn.*;
imdb            =   load(fullfile(expDir, 'iLab20M-alexnet-dagnn-catcam', 'imdb.mat'));
imdb.imageDir   =   expDir;
netObjEnv       =   load(fullfile(expDir, 'iLab20M-alexnet-dagnn-catcam', 'net-epoch-10.mat'));
netObjEnv_      =   netObjEnv.net;

netObjEnv = dagnn.DagNN.loadobj(netObjEnv_) ;
netObjEnv.accumulateParamDers   = 1;
netObjEnv.conserveMemory        = 0;
                            
modeltype    = 'dagnn';
whoseLabels  = {'object', 'environment'};

[labels_gt_obj_dagnn, labels_pred_obj_dagnn, labels_gt_env_dagnn, labels_pred_env_dagnn]  = ...
            iLab_cnn_predictBatch(netObjEnv, imdb, modeltype, whoseLabels);