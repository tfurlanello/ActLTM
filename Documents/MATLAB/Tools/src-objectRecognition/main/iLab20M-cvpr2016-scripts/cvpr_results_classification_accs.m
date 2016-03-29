% CNN classification accuracies

% alexnet
alexnets = { ...
        '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-obj/net-epoch-15.mat', ...
        '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e3/iLab20M-alexnet-dagnn-obj/net-epoch-15.mat'};

acc_alexnet = zeros(1, numel(alexnets));    
for i=1:numel(alexnets)
    load(alexnets{i});
    acc = {stats.val.error};
    acc_alexnet(i) = acc{end};
end
acc_alexnet = [acc_alexnet 0.2129]; 
% 0.2129 is the accuracy of :
% '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/older-results/iLab20M-alexnet-simplenn-cat2/net-epoch-20.mat';


% 2w-cnn-I

ww_cnn_I_nets = { ...
             '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-fc2/net-epoch-15.mat', ...
             '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-multiLevelInjection-fc2/net-epoch-14.mat', ...
             '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e3/iLab20M-alexnet-dagnn-multiLevelInjection-fc2/net-epoch-10.mat'};
         
acc_2w_cnn_I = zeros(1, numel(ww_cnn_I_nets));    
for i=1:numel(ww_cnn_I_nets)
    load(ww_cnn_I_nets{i});
    acc = {stats.val.error_obj};
    acc_2w_cnn_I(i) = acc{end};
end
         

% 2w-cnn-MI

ww_cnn_MI_net = { ...
             '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/net-epoch-11.mat', ...
             '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/net-epoch-15.mat', ...
             '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-1024/net-epoch-10.mat'};
         
acc_2w_cnn_MI = zeros(1, numel(ww_cnn_MI_net));    
for i=1:numel(ww_cnn_MI_net)
    load(ww_cnn_MI_net{i});
    acc = {stats.val.error_obj};
    acc_2w_cnn_MI(i) = acc{end};
end
                
         