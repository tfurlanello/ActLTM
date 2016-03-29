
resSaveDir = '/lab/jiaping/papers/ECCV2016/results';

resDir          =  '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2';
deCNN_folder    =  'iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000';
alexNet_folder  =  'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';


if exist(fullfile(resSaveDir, 'accuracies-iLab20M.mat'), 'file')
    return;
end


%% ============= deCNN ============================================
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

nEssentials = {'f7', 'f11', 'f18', 'f56'};

deCNN_saveDirs = { ...
    fullfile(resDir, 'f7', deCNN_folder), ...
    fullfile(resDir, 'f11', deCNN_folder), ...
    fullfile(resDir, 'f18', deCNN_folder), ...
    fullfile(resDir, 'f56', deCNN_folder)};


acc_deCNN = zeros(4,1);

for i=1:4
    load(fullfile(deCNN_saveDirs{i}, 'maxActivation/test-evalInfo.mat'));
    acc_deCNN(i) = sum(eval.gt == eval.pred(1,:,1))/numel(eval.gt);
    
end


%% ============= alexnet ============================================
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

alexnet_saveDirs = { ...
    fullfile(resDir, 'f7', alexNet_folder), ...
    fullfile(resDir, 'f11', alexNet_folder), ...
    fullfile(resDir, 'f18', alexNet_folder), ...
    fullfile(resDir, 'f56', alexNet_folder)};

acc_alexnet = zeros(4,1);
for i=1:4
    load(fullfile(alexnet_saveDirs{i}, 'maxActivation/test-evalInfo.mat'));
    acc_alexnet(i) = sum(eval.gt == eval.pred(1,:,1))/numel(eval.gt);
    
end

save(fullfile(resSaveDir, 'accuracies-iLab20M.mat'), ...
            'acc_deCNN', 'acc_alexnet', 'nEssentials');
        
        