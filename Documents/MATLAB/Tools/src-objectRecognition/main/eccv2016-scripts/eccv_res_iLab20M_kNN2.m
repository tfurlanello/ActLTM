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



%% ===================== alexnet =======================
evalfile_alexnet = fullfile(resDir, nEssentials, alexNet_folder, 'maxActivation/test-evalInfo.mat');
imdb_alexnet = load(evalfile_alexnet);
representations_alexnet = [imdb_alexnet.eval.intermediateLayers(1).value; imdb_alexnet.eval.intermediateLayers(2).value];
testIdx_alexnet = imdb_alexnet.eval.subset;
testIdxShift_alexnet = min(testIdx_alexnet)-1;
nTest = numel(testIdx_alexnet);

% euclidean distance
if ~exist(fullfile(resSaveDir, nEssentials, 'neighborsIdx-alexnet.mat'), 'file')
    neighborsIdx_alexnet = zeros(nTest, ck);
     
    parfor i=1:nTest
        if rem(i,100) == 0
            i
             
        end
        neighborsIdx_alexnet(i,:) =  knnsearch(representations_alexnet', representations_alexnet(:,i)', ...
                                'K', ck, 'Distance', 'euclidean');
%         d = dist2(representations_alexnet(:,i)', representations_alexnet');
%         [~, tmpidx] = sort(d, 'ascend');
%         neighborsIdx_alexnet(i,:) = tmpidx(1:ck);

    end
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

%{
%% =========== visualize some frames ======================
topShow = 200;
nNeighbors = 10;
qualities_diff = qualities_deCNN - qualities_alexnet;
[~, sampleIdx] = sort(qualities_diff, 'descend');
sampleIdx = sampleIdx(1:topShow);

for i=1:topShow
   refIdx = sampleIdx(i);
   tarIdx_deCNN = neighborsIdx_deCNN(refIdx, 1:nNeighbors) + testIdxShift_deCNN;
   tarIdx_alexnet = neighborsIdx_alexnet(refIdx, 1:nNeighbors) + testIdxShift_alexnet;
   
   im = [];
   for j=1:nNeighbors
       tmpIm = imread(fullfile(imdb_deCNN.imageDir, imdb_deCNN.images.name{1,tarIdx_deCNN(j)}));
       im = cat(1,im, tmpIm(:));
   end
   
   for j=1:nNeighbors
       tmpIm = imread(fullfile(imdb_alexnet.imageDir, imdb_alexnet.images.name{1,tarIdx_alexnet(j)}));
       im = cat(1,im, tmpIm(:));
   end
    
   im = imCollage(im, [256 256], [2 nNeighbors]);
   
   imwrite(im, fullfile(resSaveDir, nEssentials, ['im-' num2str(i) '.jpg']));
   
    
end


%}
