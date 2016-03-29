
resSaveDir = '/lab/igpu3/u/kai/deep_vp/results/iros2016';

saveDirs = {...
       '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc1024', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc512', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc256', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc128', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc64', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc32', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc16'};

resFileName = 'prediction.mat';
capacities = {'fc1024', 'fc512', 'fc256', 'fc128', 'fc64' , 'fc32', 'fc16'};
nCapacities = numel(capacities);

topHits = [1 2 3 4 5];
nTopHits = numel(topHits);
accuracies = zeros(nCapacities,nTopHits);


for i=1:nCapacities
    accFile = fullfile(saveDirs{i}, resFileName);
    load(accFile);
    
    gt   = labels_gt;
    pred = labels_pred;
    
    nTest = numel(gt);
    
    for j=1:nTopHits
        j_pred = pred(:,1:topHits(j));
        
        tmp_gt = repmat(gt, 1, topHits(j));
        
        cnt = sum(sum(tmp_gt == j_pred));
        
        accuracies(i,j) = cnt / nTest;
        
    end
    
end


figure;    
str = {};
for i=1:nTopHits
%     c = [rand rand rand];
%     plot(accuracies(:,i), '-o', 'markeredgecolor', c, 'linewidth', 3); hold on;
    str = cat(1, str, sprintf('top-%d accuracy', topHits(i)));
end
 
topHits = str;
save(fullfile(resSaveDir, 'network-capacity.mat'), 'accuracies', 'capacities', 'topHits');


plot(accuracies, 'linewidth', 3);
legend(str);

set(gca, 'ytick', 0:.1:1, 'xtick', 1:nCapacities, 'xticklabel', capacities, ...      
    'xgrid', 'on', 'ygrid', 'on', 'fontsize', 20);
xlim([0 nCapacities+1]);
ylim([0 1]);
xlabel('# of nodes of fc layers', 'fontsize', 25);
ylabel('accuracies of vanishing point detection', 'fontsize', 25); 
title('accuracies vs. network capacities', 'fontsize', 30);
set(gcf, 'position', [500 500 1000 700]);

export_fig(fullfile(resSaveDir, 'network-capacity.pdf'), '-pdf', '-m1', '-transparent', gcf);

