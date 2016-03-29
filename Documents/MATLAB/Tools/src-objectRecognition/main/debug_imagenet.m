

testNames = imdb.images.name;

ILSVR2010testFolder = '/lab/jiaping/igpu3home2/u/jiaping/imageNet2012/val';

for i=numel(testNames):-1:1
    i
    
    im = imread(fullfile(ILSVR2010testFolder, testNames{i}));
%     imwrite(im, fullfile('imagenet-check', testNames{i}));
    
end