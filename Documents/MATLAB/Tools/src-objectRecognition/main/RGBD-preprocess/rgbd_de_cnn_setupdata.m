function imdb = rgbd_de_cnn_setupdata(varargin)
%   iLab_setupdata_catcam  Initialize iLab20M data
%    This function creates an IMDB structure pointing to a local copy
%    of iLab20M data.
%
%    In order to speedup training and testing, it may be a good idea
%    to preprocess the images to have a fixed size (e.g. 256 pixels
%    high) and/or to store the images in RAM disk (provided that
%    sufficient RAM is available). Reading images off disk with a
%    sufficient speed is crucial for fast training.

opts.dataDir = fullfile('data','RGBD');
opts.lite = false ;
opts.rep = 4; % 4 repetitions (each image is used 4 times during each epoch)
opts = vl_argparse(opts, varargin) ;



% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------
% fid = fopen(fullfile(opts.dataDir, 'metadata-disentangling', 'train.txt'), 'r');
% traindata = textscan(fid, '%s');
% fclose(fid);
[a,b,c,d] = textread(fullfile(opts.dataDir, 'experiments', 'exp-1', ...
                                    ['train-rep' num2str(opts.rep) '.txt']), ...
                                        '%s %s %s %s\n');
traindata = {a b c d};

labelObj       =  traindata{1}; 
labelTransform =  traindata{2};

[u_labelObj, ~, labelObjIdx]             = unique(labelObj);
[u_labelTransform, ~, labelTransformIdx] = unique(labelTransform);

mapLabelObj       = containers.Map(u_labelObj, 1:numel(u_labelObj));
mapLabelTransform = containers.Map(u_labelTransform, 1:numel(u_labelTransform));

namesL = traindata{3}'; 
namesR = traindata{4}';
namesL = strcat(['scaled-256x256' filesep], namesL);
namesR = strcat(['scaled-256x256' filesep], namesR);
names  = [namesL; namesR];
nTrainImgPairs = size(names,2);

imdb.images.id      =   1:nTrainImgPairs ;
imdb.images.name    =   names ;
imdb.images.set     =   ones(1, nTrainImgPairs) ;
imdb.images.label   =   [labelObjIdx(:)'; labelTransformIdx(:)'] ;
 

imdb.classes.name       = u_labelObj ;
imdb.classes.index      = 1:numel(u_labelObj);
imdb.transforms.name    = u_labelTransform;
imdb.transforms.index   = 1:numel(u_labelTransform);
imdb.imageDir           =  opts.dataDir;
% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------
% fid = fopen(fullfile(opts.dataDir, 'metadata-disentangling', 'test.txt'), 'r');
% testdata = textscan(fid, '%s %d %s %d %s %s\n');
% fclose(fid);
[a,b,c,d] = textread(fullfile(opts.dataDir, 'experiments', 'exp-1', ...
                                    ['test-rep' num2str(opts.rep) '.txt']), ...
                                        '%s %s %s %s\n');
testdata = {a b c d};

namesL = testdata{3}'; 
namesR = testdata{4}';
namesL = strcat(['scaled-256x256' filesep], namesL);
namesR = strcat(['scaled-256x256' filesep], namesR);
names  = [namesL; namesR];
nTestImgPairs = size(names,2);

labelObj       =  testdata{1}; 
labelTransform =  testdata{2};

labelObjIdx = zeros(1, nTestImgPairs);
labelTransformIdx = zeros(1, nTestImgPairs);

for i=1:nTestImgPairs
    labelObjIdx(i) = mapLabelObj(labelObj{i});
    labelTransformIdx(i) = mapLabelTransform(labelTransform{i});
end


imdb.images.id    =  horzcat(imdb.images.id, (1:nTestImgPairs) + nTrainImgPairs) ;
imdb.images.name  =  horzcat(imdb.images.name, names) ;
imdb.images.set   =  horzcat(imdb.images.set, 3*ones(1,nTestImgPairs)) ;
imdb.images.label =  horzcat(imdb.images.label, [labelObjIdx(:)'; labelTransformIdx(:)']) ;


% -------------------------------------------------------------------------
%                                                         validation images
% -------------------------------------------------------------------------
nValidationImgs = nTestImgPairs;
rorder          = randperm(nTestImgPairs);
idxValidation   = rorder(1:nValidationImgs);

valNames            = names(:,idxValidation);
valLabelsObjIdx        = labelObjIdx(idxValidation);
valLabelsTransformIdx  = labelTransformIdx(idxValidation);

imdb.images.id    =  horzcat(imdb.images.id, (1:nValidationImgs) + nTrainImgPairs + nTestImgPairs) ;
imdb.images.name  =  horzcat(imdb.images.name, valNames) ;
imdb.images.set   =  horzcat(imdb.images.set, 2*ones(1,nValidationImgs)) ;
imdb.images.label =  horzcat(imdb.images.label, [valLabelsObjIdx(:)'; valLabelsTransformIdx(:)']) ;

end
