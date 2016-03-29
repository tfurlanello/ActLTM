function imdb = iLab_cnn_setupdata(varargin)
%   iLab_setupdata_catcam  Initialize iLab20M data
%    This function creates an IMDB structure pointing to a local copy
%    of iLab20M data.
%
%    In order to speedup training and testing, it may be a good idea
%    to preprocess the images to have a fixed size (e.g. 256 pixels
%    high) and/or to store the images in RAM disk (provided that
%    sufficient RAM is available). Reading images off disk with a
%    sufficient speed is crucial for fast training.

opts.dataDir = fullfile('data','iLab20M');
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                Load categories metadata & Training images
% -------------------------------------------------------------------------
fprintf('loading object information...\n');
flabel = fopen(fullfile(opts.dataDir, 'metadata', 'trainImageLabels.txt'), 'r');
trainLabels = textscan(flabel, '%s') ;
trainLabels = trainLabels{1};
nTrains     = numel(trainLabels);
trainLabelsCat = trainLabels(1:2:(nTrains-1));
trainLabelsCam = trainLabels(2:2:nTrains);
fclose(flabel);

[labelsObj, ~, iTrainObj] = unique(trainLabelsCat);
[labelsEnv, ~, iTrainEnv] = unique(trainLabelsCam);
objs = labelsObj';
envs = labelsEnv';

keySetObj      = labelsObj;
valueSetObj    = 1:numel(keySetObj);
mapLabelsObj   = containers.Map(keySetObj,valueSetObj); 

keySetEnv      = labelsEnv;
valueSetEnv    = 1:numel(keySetEnv);
mapLabelsEnv   = containers.Map(keySetEnv,valueSetEnv); 

descrs = [];

imdb.classes.name        =  {objs, envs} ;
imdb.classes.description =  descrs ;
imdb.imageDir            =  opts.dataDir;

% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------
ftrainImgs = fopen(fullfile(opts.dataDir, 'metadata', 'trainImageLists.txt'), 'r');
trainImgs = textscan(ftrainImgs, '%s');
trainImgs = trainImgs{1};
fclose(ftrainImgs);

ftrainImgsInfo = fullfile(opts.dataDir, 'metadata', 'trainImagesInfo.mat');
imgsinfo = load(ftrainImgsInfo);

names =  trainImgs(:)' ; 
names = strcat(['trainImages' filesep], names) ;
nTrainImgs = numel(names);

imdb.images.id      =   1:numel(names) ;
imdb.images.name    =   names ;
imdb.images.set     =   ones(1, numel(names)) ;
imdb.images.label   =   [iTrainObj'; iTrainEnv'] ;
imdb.images.info    =   imgsinfo.trainImagesInfo;
 

% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------
ftestImgs = fopen(fullfile(opts.dataDir, 'metadata', 'testImageLists.txt'), 'r');
testImgs  = textscan(ftestImgs, '%s');
testImgs  = testImgs{1};
fclose(ftestImgs);

names       =   testImgs(:)';
names       =   strcat(['testImages' filesep], names) ;
nTestImgs   =   numel(names);

ftestImgsInfo = fullfile(opts.dataDir, 'metadata', 'testImagesInfo.mat');
imgsinfo = load(ftestImgsInfo);

flabel     = fopen(fullfile(opts.dataDir, 'metadata', 'testImageLabels.txt'), 'r');
testLabels = textscan(flabel, '%s') ;
testLabels = testLabels{1};
nTest      = numel(testLabels);
testLabelsObj = testLabels(1:2:(nTest-1));
testLabelsEnv = testLabels(2:2:nTest);
fclose(flabel);

iTestLabelsObj = zeros(1, nTestImgs);
iTestLabelsEnv = zeros(1, nTestImgs);

parfor i=1:nTestImgs
    iTestLabelsEnv(i) = mapLabelsEnv(testLabelsEnv{i});
    iTestLabelsObj(i) = mapLabelsObj(testLabelsObj{i}); 
end

imdb.images.id    =  horzcat(imdb.images.id, (1:numel(names)) + nTrainImgs) ;
imdb.images.name  =  horzcat(imdb.images.name, names) ;
imdb.images.set   =  horzcat(imdb.images.set, 3*ones(1,numel(names))) ;
imdb.images.label =  horzcat(imdb.images.label, [iTestLabelsObj; iTestLabelsEnv]) ;
imdb.images.info  =  horzcat(imdb.images.info, imgsinfo.testImagesInfo);



% -------------------------------------------------------------------------
%                                                         validation images
% -------------------------------------------------------------------------
% nValidationImgs = min(2000, nTestImgs);
nValidationImgs = nTestImgs;
rorder          = randperm(nTestImgs);
idxValidation   = rorder(1:nValidationImgs);

valNames        = names(idxValidation);
valLabelsObj       = iTestLabelsObj(idxValidation);
valLabelsEnv       = iTestLabelsEnv(idxValidation);

imgsInfo = imgsinfo.testImagesInfo(idxValidation);

imdb.images.id    =  horzcat(imdb.images.id, (1:numel(valNames)) + nTrainImgs + nTestImgs) ;
imdb.images.name  =  horzcat(imdb.images.name, valNames) ;
imdb.images.set   =  horzcat(imdb.images.set, 2*ones(1,numel(valNames))) ;
imdb.images.label =  horzcat(imdb.images.label, [valLabelsObj; valLabelsEnv]) ;
imdb.images.info  =  horzcat(imdb.images.info, imgsInfo) ;


end


