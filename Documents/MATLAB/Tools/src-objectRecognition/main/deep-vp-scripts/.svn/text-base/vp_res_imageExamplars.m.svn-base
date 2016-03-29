
testImgFile = '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc1024/prediction-prob.mat';
load(testImgFile);


nTest = numel(testFiles);

imgs = [];

for i=1:46
    idx = randi(nTest);
    
    im = imread(testFiles{idx});
    
    imgs = cat(2, imgs, im(:));
    
    
end

tmp = imCollage(imgs, [300 300], [12 4]);
figure; imshow(tmp);