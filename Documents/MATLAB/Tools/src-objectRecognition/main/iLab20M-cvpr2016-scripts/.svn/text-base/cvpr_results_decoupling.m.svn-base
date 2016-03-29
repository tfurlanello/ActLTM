
saveDir = '';

maxsumFolderNames = {'maxActivation', 'sumActivation'};

%% (1) compute correlation of pose and identity entropy

alexDir = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-obj/visualization';
w2cnn_I = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-fc2/visualization'; 
w2cnn_MI = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/visualization'; 

whichLayersToEval   =   {'pool1out', 'pool2out', 'relu3out', 'relu4out', ...
                                     'pool5out', 'dropout6out', 'dropout7out'};
layerNames = {'pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7'};                                 
                                 
nEvals = numel(whichLayersToEval);            
 
% 1.1 max
corr_alex       =   zeros(nEvals,1);
corr_w2cnn_I    =   zeros(nEvals,1);
corr_w2cnn_MI   =   zeros(nEvals,1);

for i=1:nEvals
    load(fullfile(alexDir, 'maxActivation',  whichLayersToEval{i}, [whichLayersToEval{i} '.mat']));
    tmp = corrcoef(entropy_env, entropy_obj);
    corr_alex(i) = tmp(2);
    
    load(fullfile(w2cnn_I, 'maxActivation',  whichLayersToEval{i}, [whichLayersToEval{i} '.mat']));
    tmp = corrcoef(entropy_env, entropy_obj);
    corr_w2cnn_I(i) = tmp(2);
    
    load(fullfile(w2cnn_MI,'maxActivation',  whichLayersToEval{i}, [whichLayersToEval{i} '.mat']));
    tmp = corrcoef(entropy_env, entropy_obj);
    corr_w2cnn_MI(i) = tmp(2);
    
end

figure; plot(corr_alex, 'r', 'linewidth', 3); hold on;
plot(corr_w2cnn_I, 'g', 'linewidth', 3); hold on;
plot(corr_w2cnn_MI, 'b', 'linewidth', 3); hold on;
plot([0 8], [0 0]);
set(gca,'xtick', 1:nEvals, 'xticklabel', layerNames, 'fontsize' ,20);
set(gca, 'ytick', -0.8:0.2:1, 'xgrid', 'on', 'ygrid', 'on');
xlim([0 nEvals+1]);
 ylim([-0.8 1.1]);
legend({'alexnet', '2W-CNN-I', '2W-CNN-MI'}, 'fontsize', 20);
% title('what/where correlation of units on the same layer');
ylabel('correlation coefficients');
rotateXLabels(gca, 30);
set(gcf,'Position',[100 100 1000 800])

% 1.1 sum
corr_alex       =   zeros(nEvals,1);
corr_w2cnn_I    =   zeros(nEvals,1);
corr_w2cnn_MI   =   zeros(nEvals,1);

for i=1:nEvals
    load(fullfile(alexDir, 'sumActivation',  whichLayersToEval{i}, [whichLayersToEval{i} '.mat']));
    tmp = corrcoef(entropy_env, entropy_obj);
    corr_alex(i) = tmp(2);
    
    load(fullfile(w2cnn_I, 'sumActivation',  whichLayersToEval{i}, [whichLayersToEval{i} '.mat']));
    tmp = corrcoef(entropy_env, entropy_obj);
    corr_w2cnn_I(i) = tmp(2);
    
    load(fullfile(w2cnn_MI,'sumActivation',  whichLayersToEval{i}, [whichLayersToEval{i} '.mat']));
    tmp = corrcoef(entropy_env, entropy_obj);
    corr_w2cnn_MI(i) = tmp(2);
    
end

figure; plot(corr_alex, 'r', 'linewidth', 3); hold on;
plot(corr_w2cnn_I, 'g', 'linewidth', 3); hold on;
plot(corr_w2cnn_MI, 'b', 'linewidth', 3);
set(gca,'xtick', 1:nEvals, 'xticklabel', layerNames);
xlim([0 nEvals+1]);
% ylim([-1 1]);
legend({'alexnet', '2W-CNN-I', '2W-CNN-MI'});
title('sum-activation');




