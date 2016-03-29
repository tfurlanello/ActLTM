% compare of different number of camera-pairs

resSaveDir = '/lab/jiaping/papers/ECCV2016/results';
resdir  = '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/fromscratch/exp1';


% different architectures & different number of camera-pairs

arc_types = {'iLab_arc_de_dagnn_2streams_woL2', ...
              'iLab_arc_de_dagnn_2streams_wL2', ...
              'iLab_arc_de_dagnn_2streams_alexnet'};          
arc_currents     = [1  2 3];     
balancingFactors = {[1 1],  [0.05 1 1], 1};
xtickNames = {'woL2', 'wL2', 'AlexNet'};


nExp = numel(arc_currents);

repetitions = [1 2 3 4];
nReps = numel(repetitions);

errorRates = zeros(nReps, nExp);


str_legend = {};

for rep=1:nReps

	str_legend = cat(1, str_legend, ['pairs-' num2str(3*repetitions(rep))]);

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
        
        load(fullfile(ssubresDir, 'net-epoch-10.mat'));
        errorRates(rep, e) = stats.val(10).errorObject;
    end
end

accuracies = 1 - errorRates;
accuracies = accuracies';

figure; plot(accuracies, 'linewidth', 3);
set(gca, 'xtick', 1:3, 'xticklabel', xtickNames);
xlim([0.5 3.5]);
legend(str_legend);
ylabel('accuracies', 'fontsize', 20);
set(gca, 'xgrid', 'on', 'ygrid', 'on', 'fontsize', 20);
set(gcf, 'position', [200 200 1000 800]);
title('scratch: washington RGB-D object recognition', 'fontsize', 25);

% export_fig(fullfile(resSaveDir, 'rgbd-compare-scratch.pdf'), '-pdf', '-m1', '-transparent', gcf);
% export_fig(fullfile(resSaveDir, 'rgbd-compare-scratch.png'), '-png', '-m1', gcf);



