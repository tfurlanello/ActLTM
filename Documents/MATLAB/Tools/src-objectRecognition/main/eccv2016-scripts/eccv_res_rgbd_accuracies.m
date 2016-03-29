resSaveDir = '/lab/jiaping/papers/ECCV2016/results';


if exist(fullfile(resSaveDir, 'accuracies-rgbd.mat'), 'file')
    return;
end


%% (1) from scratch
reps = [1 2 3 4];
nReps = numel(reps);

nModels = 3;
modelNames = {'deCNN', 'deCNN-WO', 'alexnet'};
acc_scratch = zeros(nReps, nModels);

for e=1:nReps

    scratchRoot = ['/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/fromscratch/exp1/', ...
                        'rep-', num2str(reps(e))];

    deCNN_folderName        = 'iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000';
    deCNN_wo_folderName     = 'iLab20M-iLab_arc_de_dagnn_2streams_woL2-w1.000-w1.000';
    alexNet_folderName      = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';



    saveDirs = { ...
                fullfile(scratchRoot, deCNN_folderName), ...
                fullfile(scratchRoot, deCNN_wo_folderName), ...
                fullfile(scratchRoot, alexNet_folderName)};      
            

    for i=1:nModels
        load(fullfile(saveDirs{i}, 'maxActivation/test-evalInfo.mat'));
        acc_scratch(e,i) = sum(eval.gt == eval.pred(1,:,1))/numel(eval.gt);
    end
        
end


%% (2) from alexnet
reps = [1 2 3 4];
nReps = numel(reps);

acc_alexnet = zeros(nReps, nModels);

for e=1:nReps

    alexnetRoot = ['/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-iLab20M-AlexNet/exp1/', ...
                        'rep-', num2str(reps(e))];

    deCNN_folderName        = 'iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000';
    deCNN_wo_folderName     = 'iLab20M-iLab_arc_de_dagnn_2streams_woL2-w1.000-w1.000';
    alexNet_folderName      = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';



    saveDirs = { ...
                fullfile(alexnetRoot, deCNN_folderName), ...
                fullfile(alexnetRoot, deCNN_wo_folderName), ...
                fullfile(alexnetRoot, alexNet_folderName)};      
            

    for i=1:nModels
        load(fullfile(saveDirs{i}, 'maxActivation/test-evalInfo.mat'));
        acc_alexnet(e,i) = sum(eval.gt == eval.pred(1,:,1))/numel(eval.gt);
    end
        
end


%% (3) from deCNN
reps = [1 2 3 4];
nReps = numel(reps);

nModels = 3;
 acc_deCNN = zeros(nReps, nModels);

for e=1:nReps

    deCNNRoot = ['/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-iLab20M-deCNN/exp1/', ...
                        'rep-', num2str(reps(e))];

    deCNN_folderName        = 'iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000';
    deCNN_wo_folderName     = 'iLab20M-iLab_arc_de_dagnn_2streams_woL2-w1.000-w1.000';
    alexNet_folderName      = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';



    saveDirs = { ...
                fullfile(deCNNRoot, deCNN_folderName), ...
                fullfile(deCNNRoot, deCNN_wo_folderName), ...
                fullfile(deCNNRoot, alexNet_folderName)};      
            

    for i=1:nModels
        load(fullfile(saveDirs{i}, 'maxActivation/test-evalInfo.mat'));
        acc_deCNN(e,i) = sum(eval.gt == eval.pred(1,:,1))/numel(eval.gt);
    end
        
end


save(fullfile(resSaveDir, 'accuracies-rgbd.mat'), 'acc_deCNN', ...
                    'acc_scratch', 'acc_alexnet', 'modelNames');

