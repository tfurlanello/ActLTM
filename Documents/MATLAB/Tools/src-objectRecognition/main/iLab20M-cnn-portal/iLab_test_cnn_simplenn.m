% run trained cnn-model on test data

%==========================================================================
%                                                         working directory
%==========================================================================
expDir = '/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera';


%==========================================================================
%                                                    object prediction only
%==========================================================================
imdb    =   load(fullfile(expDir, 'iLab20M-alexnet-simplenn-cat2', 'imdb.mat'));
imdb.imageDir = expDir;
netObj  =   load(fullfile(expDir, 'iLab20M-alexnet-simplenn-cat2', 'net-epoch-20.mat'));
netObj  =   netObj.net;

inputsInfo = {'input', 'label_obj'};
modeltype = 'simplenn';
predictionNames = {};

[gt, pred] = iLab_cnn_predictBatch(netObj, imdb, modeltype, inputsInfo, predictionNames);
