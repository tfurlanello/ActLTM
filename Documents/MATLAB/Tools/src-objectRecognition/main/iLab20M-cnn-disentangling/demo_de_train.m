% portal to run CNN, based on your designed architectures
% available architecture types:
% this script is designed to train two streams 

arc_types = {'alexnet-dagnn-2streams-disentangling', ...
             'alexnet-dagnn-2streams-disentangling2', ...
              'alexnet-dagnn-2streams-disentangling3'};
          
arc_currents = [2 3 3];          
bmemory = true; 
balancingFactors = {[1 1], [0.5 1 1], [0.01 1 1]};

nExp = numel(arc_currents);
% the former architecture is built on 7-layers alexnet, 
% while the latter architecture is built on 6-layers alexnet

e = 2;
for e=1:nExp    
    arc_current = arc_currents(e);
    bFactors = balancingFactors{e};
%%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
%               setup hyperparameters for CNN                             %  
%%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

% -------------------------------------------------------------------------
%   parameters under this section are the only parameters needed to be
%   modified, and simply keep other parameters fixed
%   make sure: parameters are consistent
    fBatchNormalization     = false;
    param.modelType         = arc_types{arc_current};
    param.networkType       = 'dagnn';
    infoPortVars            = {'img1', 'img2', 'object', 'transformation'}; 
	train.derOutputs        = {'objectiveL2r',1, ...
                                    'objectiveObject', 1, 'objectiveTransformation', 1};
    
    nclasses_obj = 10;
    nclasses_transformation = 18;
    bshare = true; % two streams of alexnet share the same convolutional parameters
    bshare_lf = true;

    
% -------------------------------------------------------------------------
%                                                              dataset info
% -------------------------------------------------------------------------
    par_imdb.dataDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera';    
    par_imdb.lite = false ;
%     saveDir = par_imdb.dataDir;
    saveDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/disentangling';
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    sfx = param.modelType ;
    if fBatchNormalization
        sfx = strcat(sfx, '-bnorm') ; 
    end
    if numel(bFactors) == 3
        expDir = fullfile(saveDir, sprintf('iLab20M-%s-w%.3f-w%.3f-w%.3f', sfx, ...
                                        bFactors(1), bFactors(2), bFactors(3))) ;
    elseif numel(bFactors) == 2
        expDir = fullfile(saveDir, sprintf('iLab20M-%s-w%.3f-w%.3f', sfx, ...
                                        bFactors(1), bFactors(2))) ;
    end
    
    if exist(expDir, 'dir')
        continue;
    else
        mkdir(expDir);
    end
    
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
    train.prefetch         = false; %true ;
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
  imdb = iLab_de_cnn_setupdata('dataDir', par_imdb.dataDir, 'lite', par_imdb.lite) ;
  if ~exist(param.expDir, 'dir')
    mkdir(param.expDir) ;
  end
  save(par_imdb.imdbFile, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

switch param.modelType
    
    case 'alexnet-dagnn-2streams-disentangling'
        
        net = iLab_arc_dagnn_2streams_disentangling(nclasses_obj, nclasses_transformation,...
                                        bshare, arc, struct('isstructured', false, 'labelgraph', [])); 
                                    
        train.derOutputs = {net.outputsNames{1}, balancingFactors(1), net.outputsNames{2}, balancingFactors(2), ...
                      net.outputsNames{3}, balancingFactors(3), net.outputsNames{4},balancingFactors(4)};
        
    case 'alexnet-dagnn-2streams-disentangling2'
        
        net = iLab_arc_dagnn_2streams_disentangling2(nclasses_obj, nclasses_transformation,...
                                        bshare, bshare_lf, arc, struct('isstructured', false, 'labelgraph', [])); 
                                    
%         train.derOutputs = {net.outputsNames{1}, balancingFactors(1), net.outputsNames{2}, balancingFactors(2), ...
%                       net.outputsNames{3}, balancingFactors(3), net.outputsNames{4},balancingFactors(4)};
                  
        train.derOutputs = {net.outputsNames{1}, bFactors(1), net.outputsNames{2}, bFactors(2)};
        

    case 'alexnet-dagnn-2streams-disentangling3'
        
        net = iLab_arc_dagnn_2streams_disentangling3(nclasses_obj, nclasses_transformation,...
                                        bshare, bshare_lf, arc, struct('isstructured', false, 'labelgraph', [])); 
                                    
        train.derOutputs = {net.outputsNames{1}, bFactors(1),...
                    net.outputsNames{2}, bFactors(2),  net.outputsNames{3}, bFactors(3)};





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
        
%         net.meta.normalization.border = 256 - net.meta.normalization.imageSize(1:2) ;
        net.conserveMemory            = bmemory;
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
    fn = iLab_de_getBatchDagNNWrapper(getBatch_opts, useGpu, infoPortVars) ;
%     train = rmfield(train, {'sync', 'cudnn'}) ;
    train = rmfield(train, 'sync') ;
    info = cnn_train_dag(net, imdb, fn, train) ;
end



end