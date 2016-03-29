function imdb = ImageNet_de_cnn_setupdata(varargin)
%   iLab_setupdata_catcam  Initialize iLab20M data
%    This function creates an IMDB structure pointing to a local copy
%    of iLab20M data.
%
%    In order to speedup training and testing, it may be a good idea
%    to preprocess the images to have a fixed size (e.g. 256 pixels
%    high) and/or to store the images in RAM disk (provided that
%    sufficient RAM is available). Reading images off disk with a
%    sufficient speed is crucial for fast training.

opts.dataDir = fullfile('data','ImageNet');
opts.lite = false ;
opts.n = 40;
opts = vl_argparse(opts, varargin) ;


imdb.imageDir           =  opts.dataDir;


% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------
% fid = fopen(fullfile(opts.dataDir, 'metadata-disentangling', 'train.txt'), 'r');
% traindata = textscan(fid, '%s');
% fclose(fid);
[c,d,e,f] = textread(fullfile(opts.dataDir, 'ECCV-metadata', ['train-' num2str(opts.n) '.txt']), ...
                                        '%s %s %s %s\n');
c_num = str2double(c);
d_num = str2double(d);
traindata = {c_num d_num e f};

labelObj       =  traindata{1}; 
labelTransform =  traindata{2}; 

namesL = traindata{3}'; 
namesR = traindata{4}';
namesL = strcat(['train-crop-256' filesep], namesL);
namesR = strcat(['train-crop-256' filesep], namesR);
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
[c,d,e,f] = textread(fullfile(opts.dataDir, 'ECCV-metadata', 'test.txt'), ...
                                        '%s %s %s %s\n');
c_num = str2double(c);
d_num = str2double(d);
traindata = {c_num d_num e f};

labelObj       =  traindata{1}; 
labelTransform =  traindata{2}; 

namesL = traindata{3}'; 
namesR = traindata{4}';
namesL = strcat(['test-crop-256' filesep], namesL);
namesR = strcat(['test-crop-256' filesep], namesR);
names  = [namesL; namesR];
nTestImgPairs = size(names,2);

imdb.images.id    =  horzcat(imdb.images.id, (1:nTestImgPairs) + nTrainImgPairs) ;
imdb.images.name  =  horzcat(imdb.images.name, names) ;
imdb.images.set   =  horzcat(imdb.images.set, 3*ones(1,nTestImgPairs)) ;
imdb.images.label =  horzcat(imdb.images.label, [labelObj'; labelTransform']) ;


% -------------------------------------------------------------------------
%                                                         validation images
% -------------------------------------------------------------------------
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