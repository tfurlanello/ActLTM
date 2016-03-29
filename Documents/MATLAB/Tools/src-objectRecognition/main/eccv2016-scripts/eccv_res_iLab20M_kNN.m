% compute k nearest neighbor of each test image

resSaveDir = '/lab/jiaping/papers/ECCV2016/results';


ck = 15;
sk = 15;

nEssentials = 'f18';

resDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2';
deCNN_folder    =  'iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000';
alexNet_folder  =  'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';

% test image informations 
dataDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera/metadata';
testImgsInfoFile   =  'testImagesInfo.mat';

load(fullfile(dataDir, testImgsInfoFile));
nTest = numel(testImagesInfo);
names = fieldnames(testImagesInfo);
testImagesInfoMat = [];
for f=1:numel(names)
    testImagesInfoMat = cat(1, testImagesInfoMat, cell2mat({testImagesInfo.(names{f})}));
end
gt_objects = testImagesInfoMat(2,:);
gt_instances = testImagesInfoMat(3,:);

%% ============== deCNN ========================
evalfile_deCNN = fullfile(resDir, nEssentials, deCNN_folder, 'maxActivation/test-evalInfo.mat');
imdb_deCNN = load(evalfile_deCNN);
representations_deCNN = imdb_deCNN.eval.intermediateLayers(1).value;
testIdx_deCNN = imdb_deCNN.eval.subset;
nTest = numel(testIdx_deCNN);
testIdxShift_deCNN = min(testIdx_deCNN)-1;

% D = dist2(representations_deCNN', representations_deCNN');
% D = pdist(representations_deCNN');
% euclidean distance
if ~exist(fullfile(resSaveDir, nEssentials, 'neighborsIdx-deCNN.mat'), 'file')
    neighborsIdx_deCNN = zeros(nTest, ck);
     
    parfor i=1:nTest
        if rem(i,100) == 0
            i
             
        end
        neighborsIdx_deCNN(i,:) =  knnsearch(representations_deCNN', representations_deCNN(:,i)', ...
                                'K', ck, 'Distance', 'euclidean');
    end
    mkdir(fullfile(resSaveDir, nEssentials));
    save(fullfile(resSaveDir, nEssentials, 'neighborsIdx-deCNN.mat'), 'neighborsIdx_deCNN');                        
else
    load(fullfile(resSaveDir, nEssentials, 'neighborsIdx-deCNN.mat'));
end
                        
neighborsClass_deCNN    = gt_objects(neighborsIdx_deCNN);
neighborsInstance_deCNN = gt_instances(neighborsIdx_deCNN);

bclass = (neighborsClass_deCNN(:,2:sk) == repmat(neighborsClass_deCNN(:,1),1,(sk-1)));
binstance = (neighborsInstance_deCNN(:,2:sk) == repmat(neighborsInstance_deCNN(:,1),1,(sk-1)));
b = bclass & binstance;
qualities_deCNN = sum(b,2) / (sk-1);

 

%% ===================== alexnet =======================
evalfile_alexnet = fullfile(resDir, nEssentials, alexNet_folder, 'maxActivation/test-evalInfo.mat');
imdb_alexnet = load(evalfile_alexnet);
representations_alexnet = [imdb_alexnet.eval.intermediateLayers(1).value; imdb_alexnet.eval.intermediateLayers(2).value];
testIdx_alexnet = imdb_alexnet.eval.subset;
testIdxShift_alexnet = min(testIdx_alexnet)-1;
% euclidean distance
if ~exist(fullfile(resSaveDir, nEssentials, 'neighborsIdx-alexnet.mat'), 'file')
    neighborsIdx_alexnet =  knnsearch(representations_alexnet', representations_alexnet', ...
                                'K', ck, 'Distance', 'euclidean');
    mkdir(fullfile(resSaveDir, nEssentials));
    save(fullfile(resSaveDir, nEssentials, 'neighborsIdx-alexnet.mat'), 'neighborsIdx_alexnet');                        
else
    load(fullfile(resSaveDir, nEssentials, 'neighborsIdx-alexnet.mat'));
end

neighborsClass_alexnet    = gt_objects(neighborsIdx_alexnet);
neighborsInstance_alexnet = gt_instances(neighborsIdx_alexnet);

bclass = (neighborsClass_alexnet(:,2:sk) == repmat(neighborsClass_alexnet(:,1),1,(sk-1)));
binstance = (neighborsInstance_alexnet(:,2:sk) == repmat(neighborsInstance_alexnet(:,1),1,(sk-1)));
b = bclass & binstance;
qualities_alexnet = sum(b,2) / (sk-1);


%% =========== visualize some frames ======================
topShow = 400;
nNeighbors = 10;
qualities_diff = qualities_deCNN - qualities_alexnet;
[~, sampleIdx] = sort(qualities_diff, 'descend');
sampleIdx = sampleIdx(1:topShow);

for i=1:topShow
   refIdx = sampleIdx(i);
   tarIdx_deCNN = neighborsIdx_deCNN(refIdx, 1:nNeighbors) + testIdxShift_deCNN;
   tarIdx_alexnet = neighborsIdx_alexnet(refIdx, 1:nNeighbors) + testIdxShift_alexnet;
   
   tarIdx_deCNN = tarIdx_deCNN(1:6);
   tarIdx_alexnet = tarIdx_alexnet(2:6);
   im = [];
   for j=1:numel(tarIdx_deCNN)
       tmpIm = imread(fullfile(imdb_deCNN.imageDir, imdb_deCNN.images.name{1,tarIdx_deCNN(j)}));
       im = cat(2,im, tmpIm(:));
   end
   
   
   
   for j=1:numel(tarIdx_alexnet)
       tmpIm = imread(fullfile(imdb_alexnet.imageDir, imdb_alexnet.images.name{1,tarIdx_alexnet(j)}));
       im = cat(2,im, tmpIm(:));
   end
    
   im = imCollage(im, [256 256], [11 1]);
   
   tmpdir = fullfile(resSaveDir, nEssentials, 'good1');
   if ~exist(tmpdir, 'dir')
       mkdir(tmpdir);
   end
   
   imwrite(im, fullfile(tmpdir, ['im-' num2str(i) '.jpg']));
   
    
end



 %% =========== visualize some frames ======================
topShow = 400;
nNeighbors = 10;
qualities_diff = -qualities_deCNN + qualities_alexnet;
[~, sampleIdx] = sort(qualities_diff, 'descend');
sampleIdx = sampleIdx(1:topShow);

for i=1:topShow
   refIdx = sampleIdx(i);
   tarIdx_deCNN = neighborsIdx_deCNN(refIdx, 1:nNeighbors) + testIdxShift_deCNN;
   tarIdx_alexnet = neighborsIdx_alexnet(refIdx, 1:nNeighbors) + testIdxShift_alexnet;
   
   tarIdx_deCNN = tarIdx_deCNN(1:6);
   tarIdx_alexnet = tarIdx_alexnet(2:6);
   im = [];
   for j=1:numel(tarIdx_deCNN)
       tmpIm = imread(fullfile(imdb_deCNN.imageDir, imdb_deCNN.images.name{1,tarIdx_deCNN(j)}));
       im = cat(2,im, tmpIm(:));
   end
   
   
   
   for j=1:numel(tarIdx_alexnet)
       tmpIm = imread(fullfile(imdb_alexnet.imageDir, imdb_alexnet.images.name{1,tarIdx_alexnet(j)}));
       im = cat(2,im, tmpIm(:));
   end
    
   im = imCollage(im, [256 256], [11 1]);
   
   tmpdir = fullfile(resSaveDir, nEssentials, 'bad1');
   if ~exist(tmpdir, 'dir')
       mkdir(tmpdir);
   end
   
   imwrite(im, fullfile(tmpdir, ['im-' num2str(i) '.jpg']));
   
    
end



