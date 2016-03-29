%% two experiments

imagenet_alexnetFile = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal/pre-trained-alexnet/imagenet-matconvnet-alex.mat';  
iLab_alexnetFile  = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-obj/net-epoch-15.mat';
iLab_2wcnn_MTL_File = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/net-epoch-10.mat';


net_imageNet_alexNet = load(imagenet_alexnetFile);
net_imageNet_alexNet = dagnn.DagNN.loadobj(net_imageNet_alexNet);

net_iLab_alexnet    =   load(iLab_alexnetFile);
net_iLab_alexnet    =   dagnn.DagNN.loadobj(net_iLab_alexnet.net);

net_iLab_2WCNN_MTL  =   load(iLab_2wcnn_MTL_File);
net_iLab_2WCNN_MTL  =   dagnn.DagNN.loadobj(net_iLab_2WCNN_MTL.net);


% 2010 test images
ILSVR2010testFolder = '/home2/u/jiaping/imageNet2010/images/test';
ILSVR2010testGT     = '/home2/u/jiaping/imageNet2010/images/test-ground-truth.txt';

ims2010     = dir(fullfile(ILSVR2010testFolder, '*.JPEG')) ;
names2010   = sort({ims2010.name}) ;
labels2010  = textread(ILSVR2010testGT, '%d');
nImgs2010   = numel(names2010);

pred2010_iLab_alexnet     = zeros(1, nImgs2010);
pred2010_imageNet_alexnet = zeros(1, nImgs2010);
pred2010_iLab_2WCNN_MTL   = zeros(1, nImgs2010);

% 1. direct transfer
% evaluate the performance of our model directly on ImageNet
% evaluate alexNet (trained on ImageNet) on ImageNet

for i=1:nImgs2010
    if rem(i,100) == 0
        i
    end
    im = imread(fullfile(ILSVR2010testFolder, names2010{i}));
    pred2010_iLab_alexnet(i) = iLab_dagnn_MTL_predictSingle(net_iLab_alexnet, im);
	pred2010_imageNet_alexnet(i) = iLab_dagnn_matconvnet_predictSingle(net_imageNet_alexNet, im);
    pred = iLab_dagnn_MTL_predictSingle(net_iLab_2WCNN_MTL, im);
    pred2010_iLab_2WCNN_MTL(i) = pred(1);
end

saveDir = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-rebuttal';
save(fullfile(saveDir, 'predictions-imageNet2010.mat'), 'pred2010_iLab_2WCNN_MTL', ...
                'pred2010_iLab_alexnet', 'pred2010_imageNet_alexnet');


 