function iLab_train_complexDagNN_obj_warmstart(net_trained, saveDir, dataDir)
% train a linear-structured CNN
% instead of initializing the linear architecture by random numbers, 
% we used trained nets to initialize the linear architecture


%%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
%               setup hyperparameters for CNN                             %  
%%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

% -------------------------------------------------------------------------
%   parameters under this section are the only parameters needed to be
%   modified, and simply keep other parameters fixed
%   make sure: parameters are consistent
    fBatchNormalization     = false;
    param.arcType           = 'alex-2-2WCNN' ;
    param.networkType       = 'dagnn';
    whoseLabels             = {'object'};
    infoPortVars            = {'input', 'label_obj'};
	train.derOutputs        = {'objective_obj', 1} ;
    
    nclasses_obj = 10;
    nclasses_env = 88;    
% -------------------------------------------------------------------------
%                                                              dataset info
% -------------------------------------------------------------------------
    if ~exist('dataDir', 'var') || isempty(dataDir)
        par_imdb.dataDir = '/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera';  
    else
        par_imdb.dataDir = dataDir;
    end
    par_imdb.lite    = false ;
    
%     saveDir = '/home2/u/jiaping/iLab20M-objRec/results';

    sfx = param.arcType ;
    if fBatchNormalization
        sfx = strcat(sfx, '-bnorm') ; 
    end
    
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
    expDir = fullfile(saveDir, sprintf('iLab20M-%s', sfx)) ;
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
    arc.dropout.rate        = 0.65;
 
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
%       train.learningRate = logspace(-2, -4, 20) ;
      train.learningRate = logspace(-2.5, -4, 15) ;
    else
      train.learningRate = logspace(-1, -4, 20) ;
    end 
    train.numEpochs = numel(train.learningRate) ;
 
    param.train = train;
    
    
%%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                   run an architecture                                   %  
%%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
    
    net = iLab_arc_dagnn_fromSimpleDagNN_obj_conv1234fc2(net_trained, nclasses_obj, ...
                                nclasses_env, arc);    

    net.meta.normalization.border = [0 0];
    net.conserveMemory            = false;
    infoPortVars                  = net.inputsNames; 
    train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2}, 1} ;
    whoseLabels        = {'object', 'environment'};
    
% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
if exist(par_imdb.imdbFile, 'file')
  imdb = load(par_imdb.imdbFile) ;
else
  imdb = iLab_cnn_setupdata('dataDir', par_imdb.dataDir, 'lite', par_imdb.lite) ;
  if ~exist(param.expDir, 'dir')
    mkdir(param.expDir) ;
  end
  save(par_imdb.imdbFile, '-struct', 'imdb') ;
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
        fn = iLab_getBatchDagNNWrapper(getBatch_opts, useGpu, infoPortVars, whoseLabels) ;
    %     train = rmfield(train, {'sync', 'cudnn'}) ;
        if isfield(train, 'sync')
            train = rmfield(train, 'sync') ;
        end
        info = cnn_train_dag(net, imdb, fn, train) ;
    otherwise 
        error('Only support dagnn architectures\n');
end

end