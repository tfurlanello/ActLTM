
% plot compactness
figure ;
nObject = 10;
plot([compactness_alexnet compactness_ours]);
set(gca, 'xtick', 1:10, 'xticklabel', imdb.classes.name{1}, 'fontsize', 20);
ylabel('max-eigenvalue');
title('compactness', 'fontsize', 30);
legend({'alexnet'  'multi-layer-injection'});

% plot confusion matrix 
kmeans_confusion_alexnetMat = reshape(cell2mat(kmeans_confusion_alexnet), [nObject, nObject]);
kmeans_confusion_oursMat = reshape(cell2mat(kmeans_confusion_ours), [nObject, nObject]);


% plot activation
me = zeros(5,2);
stds = zeros(5,2);

for i=1:5
    me(i,1) =  mean(mean_activations_alexnet{i});
    stds(i,1) = std(mean_activations_alexnet{i});
    
    me(i,2) =  mean(mean_activations_ours{i});
    stds(i,2) = std(mean_activations_ours{i});
    
end

layers = {'pool1', 'pool2', 'relu3', 'relu4', 'pool5'};
for i=1:5
    figure;
    errorbar(me(i,:), stds(i,:));
    set(gca, 'xtick', [1 2], 'xticklabel', {'alexnet', 'ours'}, 'fontsize', 20);
    title([layers{i} ': mean activations'], 'fontsize', 30);
    export_fig(gcf, [layers{i} 'mean-activations.png'], '-m1');
end

% figure; plot([mean_activations_alexnet{5} mean_activations_ours{5}]);
% legend({'alexnet', 'ours'});
figure; subplot(211); plot(mean_activations_alexnet{5});
title('alexnet: mean activation of each feature channel', 'fontsize', 20);
xlim([0 numel(mean_activations_alexnet{5})+1]);
ylim([min(mean_activations_alexnet{5}) max(mean_activations_alexnet{5})]);
subplot(212); plot(mean_activations_ours{5});

title('ours: mean activation of each feature channel', 'fontsize', 20);
xlim([0 numel(mean_activations_ours{5})+1]);
ylim([min(mean_activations_alexnet{5}) max(mean_activations_alexnet{5})]);


