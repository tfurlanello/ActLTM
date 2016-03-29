% compare of different number of camera-pairs

resSaveDir = '/lab/jiaping/papers/ECCV2016/results';
resdirs  = { ...
    '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-iLab20M-AlexNet/exp1', ...
    '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-ImageNet-AlexNet/exp1', ...
    '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-iLab20M-deCNN/exp1'};

saveNames = {...
    'rgbd-compare-warmstart-iLab20M-AlexNet', ...
    'rgbd-compare-warmstart-ImageNet-AlexNet', ...
    'rgbd-compare-warmstart-iLab20M-deCNN'};
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


for w=[1 3]

    resdir = resdirs{w};
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
    accuracies = accuracies';

    figure; plot(accuracies, 'linewidth', 3);
    set(gca, 'xtick', 1:3, 'xticklabel', xtickNames);
    xlim([0.5 3.5]);
    legend(str_legend);
    ylabel('accuracies', 'fontsize', 20);
    set(gca, 'xgrid', 'on', 'ygrid', 'on', 'fontsize', 20);
    set(gcf, 'position', [200 200 1000 800]);
    title([saveNames{w} ': washington RGB-D object recognition'], 'fontsize', 15);

%     export_fig(fullfile(resSaveDir, [saveNames{w} '.pdf']), '-pdf', '-m1', '-transparent', gcf);
%     export_fig(fullfile(resSaveDir, [saveNames{w} '.png']), '-png', '-m1', gcf);


end