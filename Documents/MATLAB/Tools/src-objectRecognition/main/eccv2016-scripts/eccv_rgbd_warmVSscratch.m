% compare the performance of alexNet 
% Under (1) scratch; (2) alexNet; (3) deCNN

arc_types = {'iLab_arc_de_dagnn_2streams_woL2', ...
              'iLab_arc_de_dagnn_2streams_wL2', ...
              'iLab_arc_de_dagnn_2streams_alexnet'}; 
          
%% (1) alexnet from scratch
resdir  = '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/fromscratch/exp1';
         
arc_currents     = 3;     
balancingFactors = 1;
nExp = numel(arc_currents);

repetitions = [1 2 3 4];
nReps = numel(repetitions);
errorRates = zeros(nReps, nExp);
 
for rep=1:nReps

        arc_current = arc_currents;
        bFactors = balancingFactors;

        fBatchNormalization     = false;
        param.modelType         = arc_types{arc_current};

        
        subresDir = fullfile(resdir, ['rep-' num2str(repetitions(rep))]);

        sfx = param.modelType ;

        if numel(bFactors) == 3
            ssubresDir = fullfile(subresDir, sprintf('iLab20M-%s-w%.3f-w%.3f-w%.3f', sfx, ...
                                            bFactors(1), bFactors(2), bFactors(3))) ;
        elseif numel(bFactors) == 2
            ssubresDir = fullfile(subresDir, sprintf('iLab20M-%s-w%.3f-w%.3f', sfx, ...
                                            bFactors(1), bFactors(2))) ;
        elseif numel(bFactors) == 1
            ssubresDir = fullfile(subresDir, sprintf('iLab20M-%s', sfx));
        end
        
        load(fullfile(ssubresDir, 'net-epoch-10.mat'));
        errorRates(rep, 1) = stats.val(10).errorObject;
     
end

accuracies = 1 - errorRates;
acc_scratch = accuracies(:);


%% (2) from alexNet-iLab20M

resdir  = '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-iLab20M-AlexNet/exp1';
       
arc_currents     = 3;     
balancingFactors = 1;

nExp = numel(arc_currents);
repetitions = [1 2 3 4];
nReps = numel(repetitions);

errorRates = zeros(nReps, nExp);

for rep=1:nReps
    for e=1:nExp   
        arc_current = arc_currents;
        bFactors = balancingFactors;

        fBatchNormalization     = false;
        param.modelType         = arc_types{arc_current};


        subresDir = fullfile(resdir, ['rep-' num2str(repetitions(rep))]);

        sfx = param.modelType ;

        if numel(bFactors) == 3
            ssubresDir = fullfile(subresDir, sprintf('iLab20M-%s-w%.3f-w%.3f-w%.3f', sfx, ...
                                            bFactors(1), bFactors(2), bFactors(3))) ;
        elseif numel(bFactors) == 2
            ssubresDir = fullfile(subresDir, sprintf('iLab20M-%s-w%.3f-w%.3f', sfx, ...
                                            bFactors(1), bFactors(2))) ;
        elseif numel(bFactors) == 1
            ssubresDir = fullfile(subresDir, sprintf('iLab20M-%s', sfx));
        end

        if exist(fullfile(ssubresDir, 'net-epoch-10.mat'), 'file')
            load(fullfile(ssubresDir, 'net-epoch-10.mat'));            
            errorRates(rep, 1) = stats.val(10).errorObject;
        else
%                 errorRates(rep, e) = 0.21;
            error('file doesn''t exist\n');
        end
    end
end

accuracies = 1 - errorRates;
acc_alexnet = accuracies(:);


%% (3) from deCNN-iLab20M

resdir =  '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-iLab20M-deCNN/exp1';
        
arc_currents     = [  2 3];     
balancingFactors = { [0.05 1 1], 1};
 

nExp = numel(arc_currents);
repetitions = [1 2 3 4];
nReps = numel(repetitions);



errorRates = zeros(nReps, nExp);

for rep=1:nReps


    for e=1:nExp    

        arc_current = arc_currents(e);
        bFactors = balancingFactors{e};

        fBatchNormalization     = false;
        param.modelType         = arc_types{arc_current};


        subresDir = fullfile(resdir, ['rep-' num2str(repetitions(rep))]);

        sfx = param.modelType ;

        if numel(bFactors) == 3
            ssubresDir = fullfile(subresDir, sprintf('iLab20M-%s-w%.3f-w%.3f-w%.3f', sfx, ...
                                            bFactors(1), bFactors(2), bFactors(3))) ;
        elseif numel(bFactors) == 2
            ssubresDir = fullfile(subresDir, sprintf('iLab20M-%s-w%.3f-w%.3f', sfx, ...
                                            bFactors(1), bFactors(2))) ;
        elseif numel(bFactors) == 1
            ssubresDir = fullfile(subresDir, sprintf('iLab20M-%s', sfx));
        end

        if exist(fullfile(ssubresDir, 'net-epoch-10.mat'), 'file')
            load(fullfile(ssubresDir, 'net-epoch-10.mat'));            
            errorRates(rep, e) = stats.val(10).errorObject;
        else
%                 errorRates(rep, e) = 0.21;
            error('file doesn''t exist\n');
        end
    end
end

accuracies = 1 - errorRates;
accuracies([3 4],:) = accuracies([4 3],:);
accuracies(4,[1 2]) = accuracies(4,[2 1]);

acc_deCNN = accuracies(:,2);
acc_deCNN = acc_deCNN(:);



%% plot figure

acc = [acc_scratch acc_alexnet acc_deCNN];
str_legend = {'scratch', 'AlexNet-iLab20M', 'disCNN-iLab20M'};
nCameraPairs = {'3', '6', '9', '12'};

figure; plot(acc, 'linewidth', 3);
set(gca, 'xtick', 1:numel(nCameraPairs), 'xticklabel', nCameraPairs);
xlim([0.5 4.5]);
legend(str_legend);
ylabel('accuracies', 'fontsize', 20);
set(gca, 'xgrid', 'on', 'ygrid', 'on', 'fontsize', 20);
set(gcf, 'position', [200 200 1000 800]);
title('train AlexNet from scratch, pretrained AlexNet and pretrained disCNN', 'fontsize', 25);
    


 

