% deCNN vs AlexNet, when starting from the trained deCNN on iLab20M


%% read performances
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

acc_alexnet = accuracies(:,2);
acc_deCNN = accuracies(:,1);

%% plot figure

acc = [ acc_alexnet acc_deCNN];
str_legend = {'AlexNet', 'disCNN'};
nCameraPairs = {'3', '6', '9', '12'};

figure; plot(acc, 'linewidth', 3);
set(gca, 'xtick', 1:numel(nCameraPairs), 'xticklabel', nCameraPairs);
xlim([0.5 4.5]);
legend(str_legend);
ylabel('accuracies', 'fontsize', 20);
set(gca, 'xgrid', 'on', 'ygrid', 'on', 'fontsize', 20);
set(gcf, 'position', [200 200 1000 800]);
title('AlexNet vs disCNN', 'fontsize', 25);
    




