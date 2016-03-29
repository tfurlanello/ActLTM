% visualize relative orientation

% load('data-relativeOrientation-camins.mat');
saveDir = '/lab/jiaping/projects/iLab-object-recognition/results/check-initial-orientation';
nInstances = numel(flags);

for i=1:nInstances
    i
    selected = flags{i};
    imgNames    =   selected.imgNames;
    imgDirsIdx  =   selected.imgDirsIdx;
    imgDirs     =   selected.imgDirs;
    
    nImgs = numel(imgNames);
    
    img1d = [];
    for j=1:nImgs
        im = imread(fullfile(imgDirs{imgDirsIdx(j)}, imgNames{j}));
        img1d = cat(2, img1d, im(:));
    end
    
    bigImg = imCollage(img1d, [size(im,1), size(im,2)]);
    
    imwrite(bigImg, fullfile(saveDir, [num2str(i) '.png']));
    
end


