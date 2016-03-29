
clear all;

% (2) initialize 2W-CNN with alexnet

alexnet_modelFile = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e3/iLab20M-alexnet-dagnn-obj/net-epoch-15.mat';
load(alexnet_modelFile);
netLinear_trained = dagnn.DagNN.loadobj(net);
netLinear_trained.removeLayer('loss');
netLinear_trained.removeLayer('error');
netLinear_trained.meta.normalization.averageImage  = [] ;
netLinear_trained.meta.normalization.keepAspect    = true ;  

saveDir = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-warmstart-3';
dataDir = {};
iLab_train_complexDagNN_obj_warmstart(netLinear_trained, saveDir, dataDir);

clear all;
% (1) initialize alexnet with 2W-CNN
CNN_2W_modelFile = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/net-epoch-10.mat';
load(CNN_2W_modelFile);
net_trained = dagnn.DagNN.loadobj(net);
net_trained.meta.normalization.averageImage  = [] ;
net_trained.meta.normalization.keepAspect    = true ;  

saveDir = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-warmstart-3';
dataDir = {};
iLab_train_linearDagNN_obj_warmstart(net_trained, saveDir, dataDir);


