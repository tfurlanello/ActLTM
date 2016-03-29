   
% lights invariance training and test data setting file
lightsInvarianceData = ...
    load('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-experiments/data-lightsInvariance.mat');


iLabInfo = load('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-data-info/imagefiles-info.mat');
fid         = fopen('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-data-info/imagefiles-lists.txt', 'r');
imageNames  = textscan(fid, '%s\n');
imageNames = imageNames{1};
fclose(fid);

% test image information
labels_test     = lightsInvarianceData.labelsTest;
bTest           = lightsInvarianceData.bTest;
imageNames_test = imageNames(bTest);
nImages_test    =  sum(bTest);

testImgDir = '/lab/tmp2ig2/u/jiaping/lightsInvariance/testImages';

% model information
load('/lab/tmp2ig2/u/jiaping/lightsInvariance/iLab20M-alexnet/net-epoch-3.mat');
model = net;



labels    =  cell(nImages_test,1);
labelsIdx =  cell(nImages_test,1);
scores    =  cell(nImages_test,1);
 parfor k=1:nImages_test

    im = imread(fullfile(testImgDir, [imageFiles{k}(1:end-4) '.jpg']));
    [labels{k}, labelsIdx{k}, scores{k}] = cnnClassification(im, model);    

 end 
 