% train a bunch of CNN architectures
% a typical alexnet
% a new architecture designed by us


% portal to run CNN, based on your designed architectures
% available architecture types:
% arc_types = {'alexnet-dagnn-obj', ...
%              'alexnet-dagnn-multiLevelInjection-conv34fc2', ...
%              'alexnet-dagnn-multiLevelInjection-conv1234fc2', ...
%              'alexnet-dagnn-multiLevelInjection-fc2'};
%          
% dropoutRates = [0.5 0.5 0.65 0.5]; 
% bmemory = {true, false, false, false};
% nArcs = numel(arc_types);
% learningRates = {  logspace(-2, -3.5, 15), ...
%                     logspace(-2, -3.5, 15), ...
%                      logspace(-2.7, -4.2, 15), ...
%                       logspace(-2, -3.5, 15)};


arc_types = {'vgg_m-dagnn-obj', ...              
                'vgg_m-dagnn-multiLevelInjection-fc2', ...
                'vgg_m-dagnn-multiLevelInjection-conv34fc2'};
         
dropoutRates = [0.5   0.5  0.65]; 
bmemory = {true,  false, false};
nArcs = numel(arc_types);
learningRates = { logspace(-2, -3.5, 15), ...
                     logspace(-2, -3.5, 15), ...
                        logspace(-2.5, -4, 15)};


for a =2:nArcs
%%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
%               setup hyperparameters for CNN                             %  
%%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

% -------------------------------------------------------------------------
%   parameters under this section are the only parameters needed to be
%   modified, and simply keep other parameters fixed
%   make sure: parameters are consistent
    fBatchNormalization     = false ;
    param.arcType           = arc_types{a} ;
    param.networkType       = 'dagnn';
    whoseLabels             = {'object'} ;
    infoPortVars            = {'input', 'label_obj'} ;
	train.derOutputs        = {'objective_obj', 1} ;
    
    nclasses_obj = 10;
    nclasses_env = 88;
% -------------------------------------------------------------------------
%                                                              dataset info
% -------------------------------------------------------------------------
    par_imdb.dataDir = '/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera';    
    par_imdb.lite = false ;
    saveDir = par_imdb.dataDir;
    saveDir = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e6-vgg-m';
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    sfx = param.arcType ;
    if fBatchNormalization
        sfx = strcat(sfx, '-bnorm') ; 
    end
    expDir = fullfile(saveDir, sprintf('iLab20M-%s', sfx)) ;
	par_imdb.imdbFile = fullfile(expDir, 'imdb.mat');
    
    param.imdb = par_imdb;   
    param.expDir = expDir;
    param.trainOrderDir = saveDir;
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
    arc.dropout.rate        = dropoutRates(a);
 
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
    train.orderDir         = saveDir;

    if ~fBatchNormalization
      train.learningRate = learningRates{a};  % logspace(-2, -4, 12) ;
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
%                 make sure different architectures run on the same order 
% -------------------------------------------------------------------------
nTrain =  numel(find(imdb.images.set==1));
nepoch = 30;
rng('shuffle');
for ep=1:nepoch
    trainOrderFile = fullfile(param.trainOrderDir, sprintf('train-order-%d.mat', ep));
    if ~exist(trainOrderFile, 'file')
       trainOrder = randperm(nTrain); 
       save(trainOrderFile, 'trainOrder');
    end
end 

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

switch param.arcType
    case 'alexnet-dagnn-obj'
        net = iLab_arc_dagnn_alexnet(nclasses_obj, arc);
        train.derOutputs        = {net.outputsNames{1}, 1} ;
        whoseLabels             = {'object'};
        
    case 'vgg_m-dagnn-obj'
        net = iLab_arc_dagnn_vgg_m(nclasses_obj, arc);
        train.derOutputs        = {net.outputsNames{1}, 1} ;
        whoseLabels             = {'object'};        

    case 'alexnet-dagnn-multiLevelInjection-fc2'
        net = ...
            iLab_arc_dagnn_multiLevelInjection_fc2(nclasses_obj, nclasses_env, arc, 'alexnet', ...
                                                struct('isstructured', false, 'labelgraph', []));
        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2},1} ;
        whoseLabels        = {'object', 'environment'};
        
    case 'vgg_m-dagnn-multiLevelInjection-fc2'
        net = ...
            iLab_arc_dagnn_multiLevelInjection_fc2(nclasses_obj, nclasses_env, arc, 'vgg-m', ...
                                                struct('isstructured', false, 'labelgraph', []));
        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2},1} ;
        whoseLabels        = {'object', 'environment'};        

    case 'alexnet-dagnn-multiLevelInjection-conv1234fc2'
        net = ...
            iLab_arc_dagnn_multiLevelInjection_conv1234fc2(nclasses_obj, nclasses_env, arc, 'alexnet', ...
                                                struct('isstructured', false, 'labelgraph', []));

        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2}, 1} ;
        whoseLabels        = {'object', 'environment'};
        

    case 'vgg_m-dagnn-multiLevelInjection-conv1234fc2'
        net = ...
            iLab_arc_dagnn_multiLevelInjection_conv1234fc2(nclasses_obj, nclasses_env, arc, 'vgg-m', ...
                                                struct('isstructured', false, 'labelgraph', []));

        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2}, 1} ;
        whoseLabels        = {'object', 'environment'};        
        
    case 'alexnet-dagnn-multiLevelInjection-conv34fc2'
        net = ...
            iLab_arc_dagnn_multiLevelInjection_conv34fc2(nclasses_obj, nclasses_env, arc, 'alexnet', ...
                                                struct('isstructured', false, 'labelgraph', []));

        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2}, 1} ;
        whoseLabels        = {'object', 'environment'};       
        
    case 'vgg_m-dagnn-multiLevelInjection-conv34fc2'
        net = ...
            iLab_arc_dagnn_multiLevelInjection_conv34fc2(nclasses_obj, nclasses_env, arc, 'vgg-m', ...
                                                struct('isstructured', false, 'labelgraph', []));

        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2}, 1} ;
        whoseLabels        = {'object', 'environment'};          

    case 'alexnet-dagnn-multiLevelInjection-conv345'
        net = ...
            iLab_arc_dagnn_multiLevelInjection_conv345(nclasses_obj, nclasses_env, arc, 'alexnet', ...
                                                struct('isstructured', false, 'labelgraph', []));

        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2}, 1} ;
        whoseLabels        = {'object', 'environment'};
        
    case 'vgg_m-dagnn-multiLevelInjection-conv345'
        net = ...
            iLab_arc_dagnn_multiLevelInjection_conv345(nclasses_obj, nclasses_env, arc, 'vgg-m', ...
                                                struct('isstructured', false, 'labelgraph', []));

        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2}, 1} ;
        whoseLabels        = {'object', 'environment'};        
        
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
        
        net.meta.normalization.border = [0 0];
        net.conserveMemory            = bmemory{a};
        infoPortVars                  = net.inputsNames;        
        
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
 case 'simplenn'
    fn = iLab_getBatchSimpleNNWrapper(getBatch_opts, whoseLabels) ;
    [net,info] = cnn_train(net, imdb, fn,  train, 'conserveMemory', true) ;
  case 'dagnn'
    fn = iLab_getBatchDagNNWrapper(getBatch_opts, useGpu, infoPortVars, whoseLabels) ;
%     train = rmfield(train, {'sync', 'cudnn'}) ;
    train = rmfield(train, 'sync') ;
    info = cnn_train_dag(net, imdb, fn, train) ;
end

end