function iLab_cnnClassification_catcam(varargin)
%   This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%   VGG-VD-16, and VGG-VD-19 architectures on iLab20M data.

% run(fullfile(fileparts(mfilename('fullpath')), ...
%   '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir        = '/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera';
opts.nclasses       = 10;
opts.modelType      = 'alexnet' ;
opts.networkType    = 'dagnn' ;
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(opts.dataDir, sprintf('iLab20M-%s-%s-catcam-unstructured-cat1cam1', ...
                                       sfx, opts.networkType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize        = 128 ;
opts.train.numSubBatches    = 1 ;
opts.train.continue         = true ;
opts.train.gpus             = [1] ;
opts.train.prefetch         = true ;
opts.train.sync             = false ;
opts.train.cudnn            = true ;
opts.train.expDir           = opts.expDir ;

if ~opts.batchNormalization
  opts.train.learningRate = logspace(-2, -4, 30) ; %60
else
  opts.train.learningRate = logspace(-1, -4, 20) ;
end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = iLab_setupdata_catcam('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

labelsCam      =  imdb.classes.name{2};
interactionMat = iLab_genLabelGraphInteractionMatCR5(labelsCam);

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
net = iLab_cnnArchitecture_catcam(...
                        'model',              opts.modelType, ...
                        'batchNormalization', opts.batchNormalization, ...
                        'weightInitMethod',   opts.weightInitMethod, ...
                        'nclasses',           opts.nclasses) ;
                    
% make sure the 'category label' are exclusive;
% while the 'camera-rotation label' are inter-correlated

index = net.getParamIndex('loss_labelgraph');
net.params(index).value = [];
index = net.getParamIndex('loss_isstructured');
net.params(index).value = false;

index = net.getParamIndex('loss_camera_labelgraph');
net.params(index).value = single(interactionMat);
index = net.getParamIndex('loss_camera_isstructured');
net.params(index).value = false;

index = net.getParamIndex('error_camera_labelgraph');
net.params(index).value = single(interactionMat);
index = net.getParamIndex('error_camera_isstructured');
net.params(index).value = false;

updatelists = [];
for l=1:numel(net.layers)
    if isa(net.layers(l).block, 'dagnn.Loss') == 0
        updatelists = cat(2, updatelists, net.layers(l).paramIndexes);
    end
end
net.updatelists = updatelists;

                                                  
bopts = net.meta.normalization ;
bopts.numThreads = opts.numFetchThreads ;

% compute image statistics (mean, RGB covariances etc)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

% One can use the average RGB value, or use a different average for
% each pixel
%net.normalization.averageImage = averageImage ;
net.meta.normalization.averageImage = rgbMean ;

% switch lower(opts.networkType)
%   case 'simplenn'
%   case 'dagnn'
%     net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
%     net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
%                  {'prediction','label'}, 'top1error') ;
%   otherwise
%     error('Unknown netowrk type ''%s''.', opts.networkType) ;
% end

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

[v,d] = eig(rgbCovariance) ;
bopts.transformation = 'stretch' ;
bopts.averageImage = rgbMean ;
bopts.rgbVariance = 0.1*sqrt(d)*v' ;
useGpu = numel(opts.train.gpus) > 0 ;

switch lower(opts.networkType)
  case 'simplenn'
    fn = getBatchSimpleNNWrapper(bopts) ;
    [net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', true) ;
  case 'dagnn'
    fn = getBatchDagNNWrapper(bopts, useGpu) ;
     opts.train = rmfield(opts.train, {'sync', 'cudnn'}) ;
%    opts.train = rmfield(opts.train, {'sync'}) ;
    opts.train.derOutputs = {'objective', 1, 'objective_camera', 1} ;
    info = cnn_train_dag(net, imdb, fn, opts.train) ;
end

% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
                            'prefetch', nargout == 0) ;
labels = imdb.images.label(batch) ;

% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
                            'prefetch', nargout == 0) ;
if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  inputs = {'input', im, 'label', imdb.images.label(1,batch), ...
                         'label_camera', imdb.images.label(2,batch) } ;
end

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 101: end);
bs = 256 ;
fn = getBatchSimpleNNWrapper(opts) ;
for t=1:bs:numel(train)
  batch_time = tic ;
  batch = train(t:min(t+bs-1, numel(train))) ;
  fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
  temp = fn(imdb, batch) ;
  z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{t} = mean(temp, 4) ;
  rgbm1{t} = sum(z,2)/n ;
  rgbm2{t} = z*z'/n ;
  batch_time = toc(batch_time) ;
  fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;