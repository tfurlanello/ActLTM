% use cvpr model to visualize pose of ImageNet 2010

resSaveDir = '/lab/jiaping/papers/ECCV2016/results';
saveFolder = 'pose-estimation';
resSaveDir = fullfile(resSaveDir, saveFolder);

if ~exist(resSaveDir, 'dir')
    mkdir(resSaveDir);
end

gtfile = '/lab/igpu3/u/jiaping/imageNet2010/images/test-ground-truth.txt';
gt = dlmread(gtfile);

load('/lab/igpu3/u/jiaping/imageNet2010/results/CVPR-pose-estimation/maxActivation/test-evalInfo.mat');

refLists = [663:676 681:689  795 796 797 798];
lenLists = numel(refLists);
bRef = repmat(gt(:), 1, lenLists) == repmat(refLists(:)', numel(gt),1);
bRef = logical(sum(bRef,2));


pred_labels = labelsProb.prediction(:,1);
pred_prob   = labelsProb.probability(:,1);

% histogram

figure; hist(pred_labels);

npose = 88;

testIdx = find(evalInfo.images.set == 3);

ntop = 64;

for p=1:npose
   bflag = pred_labels == p;
   
   bflag = bflag & bRef;
   
   pIdx = find(bflag);   
   pprobs = pred_prob(pIdx);
   
   [~, idx] = sort(pprobs, 'descend');
   
   tmplen = min(ntop, numel(idx));
   idx = idx(1:tmplen);
   pIdx = pIdx(idx);
   
   pIdx = testIdx(pIdx);
   
   im = [];
   for n=1:tmplen
       
       img = imread(fullfile(evalInfo.imageDir, evalInfo.images.name{1,pIdx(n)}));
       if size(img,3) == 1
           continue;
       end
       im = cat(2, im, img(:));
       
   end
    
   if isempty(im)
       continue;
   end
       
   im = imCollage(im, [256 256], [min(10, size(im,2)) 1]);
   
   
   imwrite(im, fullfile(resSaveDir, ['im-' num2str(p) '.jpg']));
   
   
    
end




