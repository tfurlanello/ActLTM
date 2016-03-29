function imdb = vp_cnn_setupdata_2label(varargin)
%   iLab_setupdata_catcam  Initialize iLab20M data
%    This function creates an IMDB structure pointing to a local copy
%    of iLab20M data.
%
%    In order to speedup training and testing, it may be a good idea
%    to preprocess the images to have a fixed size (e.g. 256 pixels
%    high) and/or to store the images in RAM disk (provided that
%    sufficient RAM is available). Reading images off disk with a
%    sufficient speed is crucial for fast training.

opts.dataDir = 'vp';
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                Load categories metadata & Training images
% -------------------------------------------------------------------------
fprintf('loading object information...\n');
flabel = fopen(fullfile(opts.dataDir, 'trainImageLabels.txt'), 'r');
trainLabels = textscan(flabel, '%s') ;
trainLabels = trainLabels{1};
nTrains     = numel(trainLabels);
trainLabelsObj = trainLabels(1:2:(nTrains-1));
trainLabelsEnv = trainLabels(2:2:nTrains);
nTrains = nTrains/2;

fclose(flabel);

nLabels = 15;
keySetObj = cell(1,nLabels);
for i=1:nLabels
    keySetObj{i} = sprintf('%d',i);
end
keySetEnv = keySetObj;

valueSetObj    = 1:numel(keySetObj);
mapLabelsObj   = containers.Map(keySetObj,valueSetObj); 

valueSetEnv    = 1:numel(keySetEnv);
mapLabelsEnv   = containers.Map(keySetEnv,valueSetEnv); 

objs = keySetObj;
envs = keySetEnv;

iTrainLabelsObj = zeros(1, nTrains);
iTrainLabelsEnv = zeros(1, nTrains);

parfor i=1:nTrains
    iTrainLabelsEnv(i) = mapLabelsEnv(trainLabelsEnv{i});
    iTrainLabelsObj(i) = mapLabelsObj(trainLabelsObj{i}); 
end



descrs = [];

imdb.classes.name        =  {objs, envs} ;
imdb.classes.description =  descrs ;
imdb.imageDir            =  opts.dataDir;
imdb.imageDir            = '';

% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------
ftrainImgs = fopen(fullfile(opts.dataDir, 'trainImageLists.txt'), 'r');
trainImgs = textscan(ftrainImgs, '%s');
trainImgs = trainImgs{1};
fclose(ftrainImgs);
 

% names = strcat(['crop_img' filesep],  trainImgs(:)') ; 
names = trainImgs(:)';
nTrainImgs = numel(names);

imdb.images.id      =   1:numel(names) ;
imdb.images.name    =   names ;
imdb.images.set     =   ones(1, numel(names)) ;
imdb.images.label   =   [iTrainLabelsObj; iTrainLabelsEnv] ;
  

% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------
ftestImgs = fopen(fullfile(opts.dataDir,   'testImageLists.txt'), 'r');
testImgs  = textscan(ftestImgs, '%s');
testImgs  = testImgs{1};
fclose(ftestImgs);

% names       =  strcat(['crop_img' filesep],  testImgs(:)');
names = testImgs(:)';
nTestImgs   =   numel(names);
 

flabel     = fopen(fullfile(opts.dataDir, 'testImageLabels.txt'), 'r');
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
 


% -------------------------------------------------------------------------
%                                                         validation images
% -------------------------------------------------------------------------
% nValidationImgs = min(2000, nTestImgs);
nValidationImgs = nTestImgs;
rorder          = randperm(nTestImgs);
idxValidation   = rorder(1:nValidationImgs);

valNames           =    names(idxValidation);
valLabelsObj       =    iTestLabelsObj(idxValidation);
valLabelsEnv       =    iTestLabelsEnv(idxValidation);

 
imdb.images.id    =  horzcat(imdb.images.id, (1:numel(valNames)) + nTrainImgs + nTestImgs) ;
imdb.images.name  =  horzcat(imdb.images.name, valNames) ;
imdb.images.set   =  horzcat(imdb.images.set, 2*ones(1,numel(valNames))) ;
imdb.images.label =  horzcat(imdb.images.label, [valLabelsObj; valLabelsEnv]) ;
 

end