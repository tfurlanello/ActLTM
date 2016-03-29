% train an ordinary cnn and a paired cnn simultaneously
% specifically, we train them using the same initialization

% ordinary CNN - a linear-structured CNN for object recognition
% paired CNN   - a CNN architecture, with environment parameters as the
%                additional inputs

%% note: only support dagnn architecture, simplenn is not supported

% portal to run CNN, based on your designed architectures
% available architecture types:
arc_types = {'alexnet-dagnn-objenv-unstructured', ...
             'alexnet-dagnn-objenv-structured', ...
             'alexnet-dagnn-multiLevelInjection-unstructured', ...
             'alexnet-dagnn-multiLevelInjection-conv345'};

%%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
%               setup hyperparameters for CNN                             %  
%%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

% -------------------------------------------------------------------------
%   parameters under this section are the only parameters needed to be
%   modified, and simply keep other parameters fixed
%   make sure: parameters are consistent
    fBatchNormalization     = false;
    param.modelType         = 'alexnet-dagnn-multiLevelInjection-unstructured' ;
    param.networkType       = 'dagnn';
    whoseLabels             = {'object', 'environment'};
    infoPortVars            = {'input', 'label_obj', 'label_env'};
	train.derOutputs        = {'objective_obj', 1, 'objective_env', 1} ;
    
    nclasses_obj = 10;
    nclasses_env = 88;
    
    if ~strcmp(param.networkType, 'dagnn')
        fprintf(1, 'only support dagnn architecture\n');
        return;
    end
% -------------------------------------------------------------------------
%                                                              dataset info
% -------------------------------------------------------------------------
    par_imdb.dataDir = '/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera';    
    par_imdb.lite = false ;
    saveDir = par_imdb.dataDir;
    saveDir = '/home2/u/jiaping/iLab20M-objRec/results';

    sfx = param.modelType ;
    if fBatchNormalization
        sfx = strcat(sfx, '-bnorm') ; 
    end
    expDir = fullfile(saveDir, sprintf('iLab20M-%s-paired', sfx)) ;
	par_imdb.imdbFile = fullfile(expDir, 'imdb.mat');
    
    param.imdb = par_imdb;   
    param.expDir = expDir;
% -------------------------------------------------------------------------
%                                              architecture hyperparameters
% -------------------------------------------------------------------------    
    arc.batchNormalization  = fBatchNormalization ;
    arc.conv.weightInitMethod = 'gaussian';
    arc.conv.scale          = 1.0;
    arc.conv.learningRate   = [1 2];
    arc.conv.weightDecay    = [1 0];
    arc.bnorm.learningRate  = [2 1];
    arc.bnorm.weightDecay   = [10 10];
    arc.norm.param          = [5 1 0.0001/5 0.75];
    arc.pooling.method      = 'max';
    arc.dropout.rate        = 0.5;
 
    param.arc = arc;    
% -------------------------------------------------------------------------
%                                                 training hyperparameters
% -------------------------------------------------------------------------    
    train.batchSize        = 128 ;
    train.numSubBatches    = 1 ;
    train.continue         = true ;
    train.gpus             = [1] ;
    train.prefetch         = true ;
    train.sync             = false ;
    train.cudnn            = true ;
    train.expDir           = expDir ;

    if ~fBatchNormalization
      train.learningRate = logspace(-2, -4, 20) ;
    else
      train.learningRate = logspace(-1, -4, 20) ;
    end 
    train.numEpochs = numel(train.learningRate) ;
 
    param.train = train;
    
    
%%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                   run an architecture                                   %  
%%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX    

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
if exist(par_imdb.imdbFile, 'file')
  imdb = load(par_imdb.imdbFile) ;
else
  imdb = iLab_cnn_setupdata('dataDir', par_imdb.dataDir, 'lite', par_imdb.lite) ;
  mkdir(param.expDir) ;
  save(par_imdb.imdbFile, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

switch param.modelType
    case 'alexnet-dagnn-objenv-unstructured'
        [net, netLinear] = ...
            iLab_arc_dagnn_2labelLayers_objenv(nclasses_obj, nclasses_env, arc, 'alexnet', ...
                                                struct('isstructured', false, 'labelgraph', []));
        net.meta.normalization.border = [0 0];
        net.conserveMemory = false;
        
        netLinear.meta.normalization.border = [0 0];
        netLinear.conserveMemory            = false;                                                 
                                            
    case 'alexnet-dagnn-multiLevelInjection-unstructured'
        [net, netLinear] = ...
            iLab_arc_dagnn_multiLevelInjection_objenv(nclasses_obj, nclasses_env, arc, 'alexnet', ...
                                                struct('isstructured', false, 'labelgraph', []));
        net.meta.normalization.border = [0 0];
        net.conserveMemory = false;
        
        netLinear.meta.normalization.border = [0 0];
        netLinear.conserveMemory            = false;        
        
    case 'alexnet-dagnn-multiLevelInjection-conv345'
        [net, netLinear] = ...
            iLab_arc_dagnn_multiLevelInjection_conv345(nclasses_obj, nclasses_env, arc, 'alexnet', ...
                                                struct('isstructured', false, 'labelgraph', []));
        net.meta.normalization.border = [0 0];
        net.conserveMemory            = false;
        
        netLinear.meta.normalization.border = [0 0];
        netLinear.conserveMemory            = false;
        
        
    case 'alexnet-dagnn-objenv'
    case 'alexnet-dagnn-objenv-structured'
    otherwise
        error('un-recognizied model type\n');
end

%-------------------------------------------------------------------------
%       compute image statistics (mean, RGB covariances etc)
%-------------------------------------------------------------------------
switch param.networkType
    case 'simplenn'
        getBatch_opts = net.normalization ;
    case 'dagnn'
        getBatch_opts = net.meta.normalization;
end
getBatch_opts.numThreads = 12 ;
imageStatsPath = fullfile(expDir, 'imageStats.mat') ;
if exist(imageStatsPath, 'file')
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  [averageImage, rgbMean, rgbCovariance] = ...
        iLab_getImageStatistics(imdb, getBatch_opts) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

switch param.networkType
    case 'simplenn'        
        net.normalization.averageImage = rgbMean ;
    case 'dagnn'        
        net.meta.normalization.averageImage = rgbMean ;
end

% -------------------------------------------------------------------------
%  everything is prepared, run  Stochastic gradient descent
% -------------------------------------------------------------------------
[v,d] = eig(rgbCovariance) ;
getBatch_opts.transformation    = 'stretch' ;
getBatch_opts.averageImage      = rgbMean ;
getBatch_opts.rgbVariance       = 0.1*sqrt(d)*v' ;
useGpu = numel(train.gpus) > 0 ;

switch lower(param.networkType)
	case 'dagnn'
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%               train object only dagnn
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        train.expDir      = fullfile(expDir, 'simple');
        train.derOutputs  = {netLinear.outputsNames{1}, 1} ;
        whoseLabels       = {'object'};
        infoPortVars      = netLinear.inputsNames;        
        
        fn = iLab_getBatchDagNNWrapper(getBatch_opts, useGpu, infoPortVars, whoseLabels) ;
        if isfield(train, 'sync')
            train = rmfield(train, 'sync') ;
        end
%         info = cnn_train_dag(netLinear, imdb, fn, train) ;
        
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%           train complex deep nets: with object and environment labels
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        train.expDir      = fullfile(expDir, 'complex');
        train.derOutputs  = {net.outputsNames{1}, 1, net.outputsNames{2}, 1} ;
        whoseLabels       = {'object', 'environment'};
        infoPortVars      = net.inputsNames;        
        
        fn = iLab_getBatchDagNNWrapper(getBatch_opts, useGpu, infoPortVars, whoseLabels) ;
        if isfield(train, 'sync')
            train = rmfield(train, 'sync') ;
        end
        info = cnn_train_dag(net, imdb, fn, train) ;

    otherwise
        error('Only support dagnn architecture\n');
end

