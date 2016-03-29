
% setup MtConvNet in MATLAB
% run matlab/vl_setupnn

% download a pre-trained CNN from the web
% urlwrite('http://www.vlfeat.org/sandbox-matconvnet/models/imagenet-vgg-verydeep-19.mat', ...
%   'imagenet-vgg-verydeep-19.mat') ;
% net = load('imagenet-vgg-verydeep-19.mat') ;
global cnnModel;
net = load(cnnModel) ;

% obtain and preprocess an image
im = imread('tank.png') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
im_ = im_ - net.normalization.averageImage ;

% run the CNN
res = vl_simplenn(net, im_) ;

scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.classes.description{best}, best, bestScore)) ;