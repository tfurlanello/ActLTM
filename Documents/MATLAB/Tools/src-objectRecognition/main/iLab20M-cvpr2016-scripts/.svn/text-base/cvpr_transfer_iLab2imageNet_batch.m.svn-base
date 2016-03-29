%% two experiments

imagenet_alexnetFile = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal/pre-trained-alexnet/imagenet-matconvnet-alex.mat';  
iLab_alexnetFile  = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-obj/net-epoch-15.mat';
iLab_2wcnn_MTL_File = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/net-epoch-10.mat';


net_imageNet_alexNet = load(imagenet_alexnetFile);
net_imageNet_alexNet = dagnn.DagNN.loadobj(net_imageNet_alexNet);
net_imageNet_alexNet.predictionsNames = {'prediction'};
net_imageNet_alexNet.updatelists = 1:numel(net_imageNet_alexNet.params);
net_imageNet_alexNet.inputsNames = {'input' };

net_iLab_alexnet    =   load(iLab_alexnetFile);
net_iLab_alexnet    =   dagnn.DagNN.loadobj(net_iLab_alexnet.net);

net_iLab_2WCNN_MTL  =   load(iLab_2wcnn_MTL_File);
net_iLab_2WCNN_MTL  =   dagnn.DagNN.loadobj(net_iLab_2WCNN_MTL.net);


% 2010 test images
ILSVR2010test_imdb_file = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal/imdb2010test.mat';
ILSVR2012val_imdb_file = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal/imdb2012val.mat';

load(ILSVR2010test_imdb_file);
imdb2010test  = imdb;

load(ILSVR2012val_imdb_file);
imdb2012val = imdb;

% 1. direct transfer
% evaluate the performance of our model directly on ImageNet
% evaluate alexNet (trained on ImageNet) on ImageNet



saveDir = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal';

% run the trained model on the test data

%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                   parameters to get batch
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

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
 

%==========================================================================
%%                            experiment 1:  iLab_alexNet
%==========================================================================

if ~exist(fullfile(saveDir, 'imageNet-iLab_alexnet.txt'), 'file')

    fid = fopen(fullfile(saveDir, 'imageNet-iLab_alexnet.txt'), 'w');
    fprintf(fid, '1\n');
    fclose(fid);
    

    import dagnn.*;

    net_iLab_alexnet.accumulateParamDers = 1;
    net_iLab_alexnet.conserveMemory = 0;

    infoPortVars = net_iLab_alexnet.inputsNames;
    whoseLabels             = {'object'};

    opts_getbatch   = vl_argparse(dagnn_opts_getbatch, net_iLab_alexnet.meta.normalization);
    getBatch        = iLab_getBatchDagNNWrapper(opts_getbatch, 1, infoPortVars, whoseLabels) ;
    
    subset          =  find(imdb2010test.images.set==3);
    labels_iLab_alexnet_2010test = iLab_dagnn_predictBatch(net_iLab_alexnet, imdb2010test, subset, ...
                                                        getBatch, nn_opts_runTest, ...
                                                        {'predictionout'}, 'max');

    subset          =  find(imdb2012val.images.set==3);
    labels_iLab_alexnet_2012val = iLab_dagnn_predictBatch(net_iLab_alexnet, imdb2012val, subset, ...
                                                        getBatch, nn_opts_runTest, ...
                                                        {'predictionout'}, 'max');  

    save(fullfile(saveDir, 'imageNet-iLab_alexnet.mat'), 'labels_iLab_alexnet_2010test', ...
                                    'labels_iLab_alexnet_2012val');
                            
end
%==========================================================================
%%     experiment 2:                                    2W-CNN 
%==========================================================================

if ~exist(fullfile(saveDir, 'imageNet-iLab_2WCNN.txt'), 'file')
    
    fid = fopen(fullfile(saveDir, 'imageNet-iLab_2WCNN.txt'), 'w');
    fprintf(fid, '1\n');
    fclose(fid);
    
    import dagnn.*;
    net_iLab_2WCNN_MTL.accumulateParamDers = 1;
    net_iLab_2WCNN_MTL.conserveMemory = 0;

    infoPortVars = net_iLab_2WCNN_MTL.inputsNames;
    whoseLabels             = {'object', 'environment'};

    opts_getbatch   = vl_argparse(dagnn_opts_getbatch, net_iLab_2WCNN_MTL.meta.normalization);
    getBatch        = iLab_getBatchDagNNWrapper(opts_getbatch, 1, infoPortVars, whoseLabels) ;
    
    subset          =  find(imdb2010test.images.set==3);
    labels_iLab_2WCNN_2010test = iLab_dagnn_predictBatch(net_iLab_2WCNN_MTL, imdb2010test, subset, ...
                                                        getBatch, nn_opts_runTest, ...
                                                        {'predictionout'}, 'max');

    subset          =  find(imdb2012val.images.set==3);
    labels_iLab_2WCNN_2012val = iLab_dagnn_predictBatch(net_iLab_2WCNN_MTL, imdb2012val, subset, ...
                                                        getBatch, nn_opts_runTest, ...
                                                        {'predictionout'}, 'max');  

    save(fullfile(saveDir, 'imageNet-iLab_2WCNN.mat'), 'labels_iLab_2WCNN_2010test', ...
                                    'labels_iLab_2WCNN_2012val');

end
%==========================================================================
%%     experiment 3:  trained alexnet on imagenet
%==========================================================================

if ~exist(fullfile(saveDir, 'imageNet-imageNet_alexnet.txt'), 'file')
        
    fid = fopen(fullfile(saveDir, 'imageNet-imageNet_alexnet.txt'), 'w');
    fprintf(fid, '1\n');
    fclose(fid);
    
    import dagnn.*;
    net_imageNet_alexNet.accumulateParamDers = 1;
    net_imageNet_alexNet.conserveMemory = 0;

    infoPortVars = net_imageNet_alexNet.inputsNames;
    whoseLabels             = {'object'};

    opts_getbatch   = vl_argparse(dagnn_opts_getbatch, net_imageNet_alexNet.meta.normalization);
    getBatch        = iLab_getBatchDagNNWrapper(opts_getbatch, 1, infoPortVars, whoseLabels) ;
    


    subset          =  find(imdb2012val.images.set==3);
    labels_imageNet_alexnet_2012val = iLab_dagnn_predictBatch(net_imageNet_alexNet, imdb2012val, subset, ...
                                                        getBatch, nn_opts_runTest, ...
                                                        {'prediction'}, 'max'); 
                                                    
    subset          =  find(imdb2010test.images.set==3);
    labels_imageNet_alexnet_2010test = iLab_dagnn_predictBatch(net_imageNet_alexNet, imdb2010test, subset, ...
                                                        getBatch, nn_opts_runTest, ...
                                                        {'prediction'}, 'max');                                                    

    save(fullfile(saveDir, 'imageNet-imageNet_alexnet.mat'), 'labels_imageNet_alexnet_2010test', ...
                            'labels_imageNet_alexnet_2012val');

end
