function imdb = iLab_setupdata_catcam(varargin)
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
fprintf('loading category information...\n');
flabel = fopen(fullfile(opts.dataDir, 'metadata', 'trainImageLabels.txt'), 'r');
trainLabels = textscan(flabel, '%s') ;
trainLabels = trainLabels{1};
nTrains     = numel(trainLabels);
trainLabelsCat = trainLabels(1:2:(nTrains-1));
trainLabelsCam = trainLabels(2:2:nTrains);
fclose(flabel);

[labelsCat, ~, iTrainCat] = unique(trainLabelsCat);
[labelsCam, ~, iTrainCam] = unique(trainLabelsCam);
cats = labelsCat';
cams = labelsCam';

keySetCat      = labelsCat;
valueSetCat    = 1:numel(keySetCat);
mapLabelsCat   = containers.Map(keySetCat,valueSetCat); 

keySetCam      = labelsCam;
valueSetCam    = 1:numel(keySetCam);
mapLabelsCam   = containers.Map(keySetCam,valueSetCam); 

descrs = [];

imdb.classes.name        =  {cats, cams} ;
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
imdb.images.label   =   [iTrainCat'; iTrainCam'] ;
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
testLabelsCat = testLabels(1:2:(nTest-1));
testLabelsCam = testLabels(2:2:nTest);
fclose(flabel);

iTestLabelsCat = zeros(1, nTestImgs);
iTestLabelsCam = zeros(1, nTestImgs);

parfor i=1:nTestImgs
    iTestLabelsCam(i) = mapLabelsCam(testLabelsCam{i});
    iTestLabelsCat(i) = mapLabelsCat(testLabelsCat{i}); 
end

imdb.images.id    =  horzcat(imdb.images.id, (1:numel(names)) + nTrainImgs) ;
imdb.images.name  =  horzcat(imdb.images.name, names) ;
imdb.images.set   =  horzcat(imdb.images.set, 3*ones(1,numel(names))) ;
imdb.images.label =  horzcat(imdb.images.label, [iTestLabelsCat; iTestLabelsCam]) ;
imdb.images.info  =  horzcat(imdb.images.info, imgsinfo.testImagesInfo);



% -------------------------------------------------------------------------
%                                                         validation images
% -------------------------------------------------------------------------
% nValidationImgs = min(2000, nTestImgs);
nValidationImgs = nTestImgs;
rorder          = randperm(nTestImgs);
idxValidation   = rorder(1:nValidationImgs);

valNames        = names(idxValidation);
valLabelsCat       = iTestLabelsCat(idxValidation);
valLabelsCam       = iTestLabelsCam(idxValidation);

imgsInfo = imgsinfo.testImagesInfo(idxValidation);

imdb.images.id    =  horzcat(imdb.images.id, (1:numel(valNames)) + nTrainImgs + nTestImgs) ;
imdb.images.name  =  horzcat(imdb.images.name, valNames) ;
imdb.images.set   =  horzcat(imdb.images.set, 2*ones(1,numel(valNames))) ;
imdb.images.label =  horzcat(imdb.images.label, [valLabelsCat; valLabelsCam]) ;
imdb.images.info  =  horzcat(imdb.images.info, imgsInfo) ;


end


