function imdb = vp_cnn_setupdata_MTL(varargin)
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

nTasks = 6;
% -------------------------------------------------------------------------
%                                Load categories metadata & Training images
% -------------------------------------------------------------------------
fprintf('loading object information...\n');

traindata = readtable(fullfile(opts.dataDir, 'trainImageLabels.txt'), 'ReadVariableNames', false);
traindata = table2cell(traindata);


descrs = [];

imdb.classes.name        =  {} ;
imdb.classes.description =  descrs ;
imdb.imageDir            =  opts.dataDir;
imdb.imageDir            = '';
 
trainnames = traindata(:,nTasks + 1);
trainnames = trainnames';
trainLabels = traindata(:,1:nTasks);
trainLabels = cell2mat(trainLabels');

nTrainImgs = numel(trainnames);
iTrainLabels = zeros(nTasks, nTrainImgs);
%+++++++++++++++++

Label2Idx = cell(1,nTasks);

for t=1:nTasks
    tTrainLabel = trainLabels(t,:);
    label   = unique(tTrainLabel);
    nLabel  = numel(label);
    keySet  = num2cell(label);
 
    valueSet    = 1:nLabel;
    mapLabel   = containers.Map(keySet,valueSet); 
    
    Label2Idx{t} = mapLabel;
    
    parfor i=1:nTrainImgs
        iTrainLabels(t,i) = mapLabel(tTrainLabel(i));
    end

end
% +++++++++++++++++++


imdb.Label2Idx      = Label2Idx;
imdb.images.id      =   1:numel(trainnames) ;
imdb.images.name    =   trainnames ;
imdb.images.set     =   ones(1, numel(trainnames)) ;
imdb.images.label   =   iTrainLabels;
  

% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------
testdata = readtable(fullfile(opts.dataDir, 'testImageLabels.txt'), 'ReadVariableNames', false);
testdata = table2cell(testdata);

testnames = testdata(:,nTasks + 1);
testnames = testnames';
testLabels = testdata(:,1:nTasks);
testLabels = cell2mat(testLabels');

nTestImgs = numel(testnames);
iTestLabels = zeros(nTasks, nTestImgs);
%% ==========
for t=1:nTasks
    tTestLabel = testLabels(t,:);
    
    mapLabel   = Label2Idx{t};
    
    parfor i=1:nTestImgs
        iTestLabels(t,i) = mapLabel(tTestLabel(i));
    end

end
%% =============

imdb.images.id    =  horzcat(imdb.images.id, (1:numel(testnames)) + nTrainImgs) ;
imdb.images.name  =  horzcat(imdb.images.name, testnames) ;
imdb.images.set   =  horzcat(imdb.images.set, 3*ones(1,numel(testnames))) ;
imdb.images.label =  horzcat(imdb.images.label, iTestLabels) ;
 


% -------------------------------------------------------------------------
%                                                         validation images
% -------------------------------------------------------------------------
% nValidationImgs = min(2000, nTestImgs);
nValidationImgs = nTestImgs;
rorder          = randperm(nTestImgs);
idxValidation   = rorder(1:nValidationImgs);

valNames           =    testnames(idxValidation);
valLabels          =    testLabels(:, idxValidation);

 
imdb.images.id    =  horzcat(imdb.images.id, (1:numel(valNames)) + nTrainImgs + nTestImgs) ;
imdb.images.name  =  horzcat(imdb.images.name, valNames) ;
imdb.images.set   =  horzcat(imdb.images.set, 2*ones(1,numel(valNames))) ;
imdb.images.label =  horzcat(imdb.images.label, valLabels) ;
 

end