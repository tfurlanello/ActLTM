% portal to run CNN, based on your designed architectures
% available architecture types:
arc_types = {'alexnet-simplenn-obj',  'alexnet-simplenn-env', ...
             'vgg-m-simplenn-obj',    'vgg-m-simplenn-env', ...
             'alexnet-dagnn-obj',     'alexnet-dagnn-env', ...
             'vgg-m-dagnn-obj',       'vgg-m-dagnn-env', ...
             'alexnet-dagnn-objenv-unstructured',  'alexnet-dagnn-objenv-structured', ...
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
    param.modelType         = 'alexnet-dagnn-multiLevelInjection-fc2'; % 'alexnet-dagnn-obj' ;
    param.networkType       = 'dagnn';
    whoseLabels             = {'object'};
    infoPortVars            = {'input', 'label_obj'};
	train.derOutputs        = {'objective_obj', 1} ;
    
    nclasses_obj = 10;
    nclasses_env = 88;
    b_noise_injection = true;
% -------------------------------------------------------------------------
%                                                              dataset info
% -------------------------------------------------------------------------
    par_imdb.dataDir = '/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera';    
    par_imdb.lite = false ;
    saveDir = par_imdb.dataDir;
    saveDir = '/home2/u/jiaping/iLab20M-objRec/results/MTL-noise-label-injection';
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    sfx = param.modelType ;
    if fBatchNormalization
        sfx = strcat(sfx, '-bnorm') ; 
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
      train.learningRate = logspace(-2, -4, 12) ;
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
    case 'alexnet-simplenn-obj'
        net = iLab_simplenn_alexnet2(nclasses_obj, arc);
	case 'alexnet-simplenn-env'
        net = iLab_simplenn_alexnet2(nclasses_env, arc);
    case 'vgg-m-simplenn-obj'
        net = iLab_simplenn_vgg_m2(nclasses_obj, arc);
	case 'vgg-m-simplenn-env'
        net = iLab_simplenn_vgg_m2(nclasses_env, arc);
    case 'alexnet-dagnn-obj'
        net = iLab_arc_dagnn_alexnet(nclasses_obj, arc);

        train.derOutputs        = {net.outputsNames{1}, 1} ;
        whoseLabels             = {'object'};

    case 'alexnet-dagnn-env'
        net = iLab_arc_dagnn_alexnet(nclasses_env, arc);
    case 'vgg-m-dagnn-obj'
        net = iLab_arc_dagnn_vgg_m(nclasses_obj, arc);
	case 'vgg-m-dagnn-env'
        net = iLab_arc_dagnn_vgg_m(nclasses_env, arc);  
    case 'alexnet-dagnn-multiLevelInjection-fc2'
%         net = ...
%             iLab_arc_dagnn_2labelLayers_objenv(nclasses_obj, nclasses_env, arc, 'alexnet', ...
%                                                 struct('isstructured', false, 'labelgraph', []));
        net = ...
            iLab_arc_dagnn_multiLevelInjection_fc2(nclasses_obj, nclasses_env, arc, 'alexnet', ...
                                                struct('isstructured', false, 'labelgraph', []));
        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2},1} ;
        whoseLabels        = {'object', 'environment'};

    case 'alexnet-dagnn-multiLevelInjection-conv1234fc2'
        net = ...
            iLab_arc_dagnn_multiLevelInjection_conv1234fc2(nclasses_obj, nclasses_env, arc, 'alexnet', ...
                                                struct('isstructured', false, 'labelgraph', []));

        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2}, 1} ;
        whoseLabels        = {'object', 'environment'};
        
    case 'alexnet-dagnn-multiLevelInjection-conv34fc2'
        net = ...
            iLab_arc_dagnn_multiLevelInjection_conv34fc2(nclasses_obj, nclasses_env, arc, 'alexnet', ...
                                                struct('isstructured', false, 'labelgraph', []));

        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2}, 1} ;
        whoseLabels        = {'object', 'environment'};        

    case 'alexnet-dagnn-multiLevelInjection-conv345'
        net = ...
            iLab_arc_dagnn_multiLevelInjection_conv345(nclasses_obj, nclasses_env, arc, 'alexnet', ...
                                                struct('isstructured', false, 'labelgraph', []));

        train.derOutputs   = {net.outputsNames{1}, 1, net.outputsNames{2}, 1} ;
        whoseLabels        = {'object', 'environment'};
        
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
        
        net.meta.normalization.border = [0 0];
        net.conserveMemory            = false;
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


% -------------------------------------------------------------------------
%  multi-task learning: inject auxiliary labels (random noise)
% -------------------------------------------------------------------------
if b_noise_injection
    env_labels   =  imdb.images.label(2,:);
    u_env_labels =  unique(env_labels);
    nEnvLabels   =  numel(u_env_labels);
    minl = min(u_env_labels);
    maxl = max(u_env_labels);
    
    env_labels_copy = env_labels;
    for l=1:nEnvLabels
        lidx = find(env_labels_copy == u_env_labels(l));
        env_labels(lidx) = randi([minl, maxl], 1, numel(lidx));
    end
    imdb.images.label(2,:) = env_labels;    
end



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



