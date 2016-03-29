function imdb = iLab_de_cnn_setupdata_tmp(varargin)
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
opts.nf = 18;
opts = vl_argparse(opts, varargin) ;

metaDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/test-imdb';
% -------------------------------------------------------------------------
%                                                             label mapping
% -------------------------------------------------------------------------
load(fullfile(metaDir, ['mappings-f' num2str(opts.nf) '.mat']));
objs          =  mapObject.keys;
objsIdx       =  mapObject.values;
transforms    =  mapTransform.keys;
transformsIdx =  mapTransform.values;

imdb.classes.name       =  objs ;
imdb.classes.index      =  objsIdx;
imdb.transforms.name    = transforms;
imdb.transforms.index   = transformsIdx;
imdb.imageDir           =  opts.dataDir;


% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------
% fid = fopen(fullfile(opts.dataDir, 'metadata-disentangling', 'train.txt'), 'r');
% traindata = textscan(fid, '%s');
% fclose(fid);
[a,b,c,d,e,f] = textread(fullfile(metaDir, ['train-f' num2str(opts.nf) '.txt']), ...
                                        '%s %s %s %s %s %s\n');
b_num = str2double(b);
d_num = str2double(d);
traindata = {a b_num c d_num e f};

labelObj       =  traindata{2}; 
labelTransform =  traindata{4}; 

namesL = traindata{5}'; 
namesR = traindata{6}';
namesL = strcat(['trainImages' filesep], namesL);
namesR = strcat(['trainImages' filesep], namesR);
names  = [namesL; namesR];
nTrainImgPairs = size(names,2);

imdb.images.id      =   1:nTrainImgPairs ;
imdb.images.name    =   names ;
imdb.images.set     =   ones(1, nTrainImgPairs) ;
imdb.images.label   =   [labelObj'; labelTransform'] ;
 
% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------
% fid = fopen(fullfile(opts.dataDir, 'metadata-disentangling', 'test.txt'), 'r');
% testdata = textscan(fid, '%s %d %s %d %s %s\n');
% fclose(fid);
[a,b,c,d,e,f] = textread(fullfile(metaDir, ['test-f' num2str(opts.nf) '.txt']), ...
                                        '%s %s %s %s %s %s\n');
b_num = str2double(b);
d_num = str2double(d);
testdata = {a b_num c d_num e f};

labelObj       =  testdata{2}; 
labelTransform =  testdata{4}; 

namesL = testdata{5}'; 
namesR = testdata{6}';
namesL = strcat(['testImages' filesep], namesL);
namesR = strcat(['testImages' filesep], namesR);
names  = [namesL; namesR];
nTestImgPairs = size(names,2);

imdb.images.id    =  horzcat(imdb.images.id, (1:nTestImgPairs) + nTrainImgPairs) ;
imdb.images.name  =  horzcat(imdb.images.name, names) ;
imdb.images.set   =  horzcat(imdb.images.set, 3*ones(1,nTestImgPairs)) ;
imdb.images.label =  horzcat(imdb.images.label, [labelObj'; labelTransform']) ;


% -------------------------------------------------------------------------
%                                                         validation images
% -------------------------------------------------------------------------
% nValidationImgs = nTestImgPairs;
% rorder          = randperm(nTestImgPairs);
% idxValidation   = rorder(1:nValidationImgs);
% 
% valNames            = names(:,idxValidation);
% valLabelsObj        = labelObj(idxValidation);
% valLabelsTransform  = labelTransform(idxValidation);
% 
% imdb.images.id    =  horzcat(imdb.images.id, (1:nValidationImgs) + nTrainImgPairs + nTestImgPairs) ;
% imdb.images.name  =  horzcat(imdb.images.name, valNames) ;
% imdb.images.set   =  horzcat(imdb.images.set, 2*ones(1,nValidationImgs)) ;
% imdb.images.label =  horzcat(imdb.images.label, [valLabelsObj'; valLabelsTransform']) ;


nValidationImgs = nTestImgPairs;
rorder          = randperm(nTestImgPairs);
idxValidation   = rorder(1:nValidationImgs);
idxValidation   = 1:1000:nValidationImgs;
nValidationImgs = numel(idxValidation);

valNames            = names(:,idxValidation);
valLabelsObj        = labelObj(idxValidation);
valLabelsTransform  = labelTransform(idxValidation);

imdb.images.id    =  horzcat(imdb.images.id, (1:nValidationImgs) + nTrainImgPairs + nTestImgPairs) ;
imdb.images.name  =  horzcat(imdb.images.name, valNames) ;
imdb.images.set   =  horzcat(imdb.images.set, 2*ones(1,nValidationImgs)) ;
imdb.images.label =  horzcat(imdb.images.label, [valLabelsObj'; valLabelsTransform']) ;


end