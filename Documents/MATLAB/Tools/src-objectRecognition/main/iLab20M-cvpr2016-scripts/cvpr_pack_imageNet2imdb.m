%  package imageNet images to imdb format

saveDir = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal';

%% 2010 test images
ILSVR2010testFolder = '/home2/u/jiaping/imageNet2010/images/test';
ILSVR2010testGT     = '/home2/u/jiaping/imageNet2010/images/test-ground-truth.txt';

ims2010     = dir(fullfile(ILSVR2010testFolder, '*.JPEG')) ;
names2010   = sort({ims2010.name}) ;
labels2010  = textread(ILSVR2010testGT, '%d');
nImgs2010   = numel(names2010);

names2010 = names2010(nImgs2010:-1:1);
labels2010 = labels2010(nImgs2010:-1:1);

imdb.imageDir            =  ILSVR2010testFolder;
names =  names2010(:)' ; 
imdb.images.id      =   1:numel(names) ;
imdb.images.name    =   names ;
imdb.images.set     =   ones(1, numel(names)) + 2 ;
imdb.images.label   =   [ones(1, numel(names)); ones(1, numel(names))] ;
imdb.images.gtlabel   =   [labels2010(:)'; labels2010(:)'] ;


save(fullfile(saveDir, 'imdb2010test.mat'), 'imdb');

%% 2012 test images
ILSVR2012valFolder = '/home2/u/jiaping/imageNet2012/val';
ILSVR2012valGT     = '/home2/u/jiaping/imageNet2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt';

ims2012     = dir(fullfile(ILSVR2012valFolder, '*.JPEG')) ;
names2012   = sort({ims2012.name}) ;
labels2012  = textread(ILSVR2012valGT, '%d');
nImgs2012   = numel(names2012);

names2012  = names2012(nImgs2012:-1:1);
labels2012 = labels2012(nImgs2012:-1:1);



imdb.imageDir            =  ILSVR2012valFolder;
names =  names2012(:)' ; 
imdb.images.id      =   1:numel(names) ;
imdb.images.name    =   names ;
imdb.images.set     =   ones(1, numel(names)) + 2;
imdb.images.label   =   [ones(1, numel(names)); ones(1, numel(names))]  ;
imdb.images.gtlabel   =   [labels2012(:)'; labels2012(:)']  ;

save(fullfile(saveDir, 'imdb2012val.mat'), 'imdb');