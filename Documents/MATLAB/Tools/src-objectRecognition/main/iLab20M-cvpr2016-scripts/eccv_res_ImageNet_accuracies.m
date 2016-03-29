resSaveDir = '/lab/jiaping/papers/ECCV2016/results';

if exist(fullfile(resSaveDir, 'accuracies-ImageNet.mat'), 'file')
    return;
end


%% ============= scratch ============================================
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

scratchRoot = '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/fromscratch';
arcFolder = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';

    
alexnet_saveDirs = { ...
               fullfile(scratchRoot, 'n5', arcFolder), ...
               fullfile(scratchRoot, 'n10', arcFolder), ...
               fullfile(scratchRoot, 'n20', arcFolder), ...
               fullfile(scratchRoot, 'n40', arcFolder)};
           
nTrains = {'n5', 'n10', 'n20', 'n40'};
nCases = numel(nTrains);
acc_scratch_top1 = zeros(nCases,1);
acc_scratch_top5 = zeros(nCases,1);

for i=1:nCases
    load(fullfile(alexnet_saveDirs{i}, 'maxActivation/test-evalInfo.mat'));
    acc_scratch_top1(i) = sum(eval.gt == eval.pred(1,:,1))/numel(eval.gt);
    
    i_gt = eval.gt;
    i_gt = i_gt(:);
    i_pred = squeeze(eval.pred);
    
    b = i_pred == repmat(i_gt,1,5);
    acc_scratch_top5(i) = sum(sum(b,2))/numel(i_gt);
    
end
 

%% ============= pretrained alexnet ============================================
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

alexnetRoot = '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/warmstart-iLab20M-AlexNet-4096';
arcFolder = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';

    
alexnet_saveDirs = { ...
               fullfile(alexnetRoot, 'n5', arcFolder), ...
               fullfile(alexnetRoot, 'n10', arcFolder), ...
               fullfile(alexnetRoot, 'n20', arcFolder), ...
               fullfile(alexnetRoot, 'n40', arcFolder)};
           
acc_alexnet_top1 = zeros(nCases,1);
acc_alexnet_top5 = zeros(nCases,1);

for i=1:nCases
    load(fullfile(alexnet_saveDirs{i}, 'maxActivation/test-evalInfo.mat'));
    acc_alexnet_top1(i) = sum(eval.gt == eval.pred(1,:,1))/numel(eval.gt);
    
    i_gt = eval.gt;
    i_gt = i_gt(:);
    i_pred = squeeze(eval.pred);
    
    b = i_pred == repmat(i_gt,1,5);
    acc_alexnet_top5(i) = sum(sum(b,2))/numel(i_gt);
    
end      



%% ============= pretrained deCNN ============================================
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

deCNNRoot = '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/warmstart-iLab20M-deCNN-4096';
arcFolder = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';

    
alexnet_saveDirs = { ...
               fullfile(deCNNRoot, 'n5', arcFolder), ...
               fullfile(deCNNRoot, 'n10', arcFolder), ...
               fullfile(deCNNRoot, 'n20', arcFolder), ...
               fullfile(deCNNRoot, 'n40', arcFolder)};

acc_deCNN_top1 = zeros(nCases,1);
acc_deCNN_top5 = zeros(nCases,1);

for i=1:nCases
    load(fullfile(alexnet_saveDirs{i}, 'maxActivation/test-evalInfo.mat'));
    acc_deCNN_top1(i) = sum(eval.gt == eval.pred(1,:,1))/numel(eval.gt);
    
    i_gt = eval.gt;
    i_gt = i_gt(:);
    i_pred = squeeze(eval.pred);
    
    b = i_pred == repmat(i_gt,1,5);
    acc_deCNN_top5(i) = sum(sum(b,2))/numel(i_gt);
    
end      

        
save(fullfile(resSaveDir, 'accuracies-ImageNet.mat'), ...
            'acc_scratch_top1', 'acc_scratch_top5', ...
            'acc_alexnet_top1', 'acc_alexnet_top5', ...
            'acc_deCNN_top1', 'acc_deCNN_top5', 'nTrains');
        
           