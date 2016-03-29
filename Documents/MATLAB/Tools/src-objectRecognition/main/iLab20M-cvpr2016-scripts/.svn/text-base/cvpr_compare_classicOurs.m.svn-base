% this script is used to compare classic alexnet and our architecture

resDir =  '/lab/ilab/30/jiaping/iLab20M-objectRecognition';

nObject = 10;
nEnv = 88;
%% architecture files
netFileAlexnet = fullfile(resDir, 'alexnet-19.mat');
netFileOurs = fullfile(resDir, 'oursnet-10.mat');

%% evaluation files
evalFileAlexnet = fullfile(resDir, 'imdb-sumEval-alexnet.mat');
evalFileOurs = fullfile(resDir, 'imdb-sumEval-ours.mat');

%% save info
saveDir =  '/lab/ilab/30/jiaping/iLab20M-objectRecognition';
meActivationFileAlexnet = 'meanActivationsAlexnet.mat';
meActivationFileOurs = 'meanActivationsOurs.mat';

compactnessFileOurs = 'compactnessOurs.mat';
compactnessFileAlexnet = 'compactnessAlexnet.mat';

kmeansConfusionFileAlexnet = 'kmeansConfusionAlexnet.mat';
kmeansConfusionFileOurs = 'kmeansConfusionOurs.mat';

filtMagFileAlexnet = 'filtMagAlexnet.mat';
filtMagFileOurs = 'filtMagOurs.mat';

%% 1. alexnet: check the sum activations for each filter
whichLayersToEval   =   {'pool1out', 'pool2out', 'relu3out', 'relu4out', 'pool5out', 'dropout7out'};

if ~exist('evalAlexnet', 'var')
    imdb_alexnet = load(evalFileAlexnet);
    evalAlexnet = imdb_alexnet.eval;
    clear imdb_alexnet;
end
nIntermediateLayers = numel(whichLayersToEval);
assert( numel(evalAlexnet.intermediateLayers) == nIntermediateLayers);

if ~exist(fullfile(saveDir, meActivationFileAlexnet), 'file')
    % mean response
    mean_activations_alexnet = cell(numel(whichLayersToEval)-1,1);
    for l=1:(nIntermediateLayers-1)
        layerName   =   evalAlexnet.intermediateLayers(l).name;
        values      =   evalAlexnet.intermediateLayers(l).value;
        subset      =   evalAlexnet.subset;
        nchannels   =   size(values,1);
        mean_activations_alexnet{l} = mean(values,2);
    end
    save(fullfile(saveDir, meActivationFileAlexnet), 'mean_activations_alexnet');
end

% compute the magnitude of filters
net_linear = load(netFileAlexnet);
net_linear = net_linear.net;
net_linear = dagnn.DagNN.loadobj(net_linear);
layerNames = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};
nLayers = numel(layerNames);

if ~exist(fullfile(saveDir, filtMagFileAlexnet), 'file')
    filtMags = cell(nLayers,1);
    for l=1:nLayers
       params =  net_linear.layers(net_linear.getLayerIndex(layerNames{l})).params;
       filt = params{1};
       filtMag = net_linear.params(net_linear.getParamIndex(filt)).value;
       filtMag = filtMag .* filtMag;
       
       filtMag = reshape(filtMag, [size(filtMag,1)*size(filtMag,2)*size(filtMag,3) size(filtMag,4)]);
       filtMag = sqrt(sum(filtMag,1));
       filtMags{l} = filtMag(:);
    end
   save(fullfile(saveDir, filtMagFileAlexnet), 'filtMags');
    
end



% compute compactness of each class
values = evalAlexnet.intermediateLayers(nIntermediateLayers).value;
fc2          =  values';
label_gt     =  evalAlexnet.gt;

if ~exist(fullfile(saveDir, compactnessFileAlexnet), 'file')
    compactness_alexnet = zeros(nObject,1);
    for c=1:nObject
        ind = (label_gt == c);
        X = fc2(ind,:);
        s = svd(X);
        compactness_alexnet(c) = s(1);
    end
    save(fullfile(saveDir, compactnessFileAlexnet), 'compactness_alexnet');
    
end


% see how k-means works on the fc2 representation

if ~exist(fullfile(saveDir, kmeansConfusionFileAlexnet), 'file')    
    sparseIdx = 1:10:size(fc2,1);
    label_gt = label_gt(sparseIdx);
    fc2 = fc2(sparseIdx, :);
    
    idx_clusters =  kmeans(fc2, nObject);

    kmeans_confusion_alexnet = cell(nObject,1);
    for clust=1:nObject
        ind = idx_clusters == clust;
        ind_label = label_gt(ind);
        counts = hist(ind_label, 1:1:nObject);
        kmeans_confusion_alexnet{clust} = counts;
    end
    save(fullfile(saveDir, kmeansConfusionFileAlexnet), 'kmeans_confusion_alexnet');
end



%% 2. ours: check the sum activations for each filter 
whichLayersToEval   =  {'pool1out', 'pool2out', 'relu3out', 'relu4out', 'pool5out', 'dropout7out', ...
                             'conv32dropoutout', 'conv42dropoutout'};

if ~exist('evalOurs', 'var')                         
    imdb_ours = load(evalFileOurs);
    evalOurs = imdb_ours.eval;
    clear imdb_ours;
end

nIntermediateLayers = numel(whichLayersToEval);
assert( numel(evalOurs.intermediateLayers) == nIntermediateLayers);

% mean response
if ~exist(fullfile(saveDir, meActivationFileOurs), 'file')
    mean_activations_ours = cell(numel(whichLayersToEval)-1,1);
    for l=1:(nIntermediateLayers-3)
        layerName   =   evalOurs.intermediateLayers(l).name;
        values      =   evalOurs.intermediateLayers(l).value;
        subset      =   evalOurs.subset;
        nchannels   =   size(values,1);
        mean_activations_ours{l} = mean(values,2);
    end
    save(fullfile(saveDir, meActivationFileOurs), 'mean_activations_ours');
end


% compute the magnitude of filters
net_ours = load(netFileAlexnet);
net_ours = net_ours.net;
net_ours = dagnn.DagNN.loadobj(net_ours);
layerNames = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};
nLayers = numel(layerNames);

if ~exist(fullfile(saveDir, filtMagFileOurs), 'file')
    filtMags = cell(nLayers,1);
    for l=1:nLayers
       params =  net_ours.layers(net_ours.getLayerIndex(layerNames{l})).params;
       filt = params{1};
       filtMag = net_ours.params(net_ours.getParamIndex(filt)).value;
       filtMag = filtMag .* filtMag;
       
       filtMag = reshape(filtMag, [size(filtMag,1)*size(filtMag,2)*size(filtMag,3) size(filtMag,4)]);
       filtMag = sqrt(sum(filtMag,1));
       filtMags{l} = filtMag(:);
    end
   save(fullfile(saveDir, filtMagFileOurs), 'filtMags');
    
end

% compute compactness of each class
values = evalOurs.intermediateLayers(nIntermediateLayers-2).value;
fc2          =  values';
label_gt     =  evalOurs.gt(1,:);

if ~exist(fullfile(saveDir, compactnessFileOurs), 'file')
    compactness_ours = zeros(nObject,1);
    for c=1:nObject
        ind = (label_gt == c);
        X = fc2(ind,:);
        s = svd(X);
        compactness_ours(c) = s(1);
    end
    save(fullfile(saveDir, compactnessFileOurs), 'compactness_ours');
end

% see how k-means works on the fc2 representation

if ~exist(fullfile(saveDir, kmeansConfusionFileOurs), 'file')
    
	sparseIdx = 1:10:size(fc2,1);
    label_gt = label_gt(sparseIdx);
    fc2 = fc2(sparseIdx, :);
    
    idx_clusters =  kmeans(fc2, nObject);
    kmeans_confusion_ours = cell(nObject,1);
    for clust=1:nObject
        ind = idx_clusters == clust;
        ind_label = label_gt(ind);
        counts = hist(ind_label, 1:1:nObject);
        kmeans_confusion_ours{clust} = counts;
    end
    save(fullfile(saveDir, kmeansConfusionFileOurs), 'kmeans_confusion_ours');

end

% confusion matrix 
layerNames = { 'dropout7out', ...
                             'conv32dropoutout', 'conv42dropoutout'};

predvalues = 0;                         
for l=1:numel(layerNames)   
complexEvalfcE = load(fullfile(resDir, [layerNames{l} '-pred-evn.mat']));
[~,pred] = max(complexEvalfcE.(layerNames{l}), [], 2);
gt = complexEvalfcE.gt(2,:);
acc = sum(pred(:) == gt(:))/numel(gt);
figure;
 plotCM(gt ,pred, classes.name{2});
 title([layerNames{l} '-accuracy: ' num2str(acc)], 'fontsize' ,20);
 
 predvalues = predvalues + complexEvalfcE.(layerNames{l});
 set (gcf, 'Units', 'normalized', 'Position', [0,0,1,1]);
 export_fig(gcf, [layerNames{l} '.png'], '-m1');
 close all;
end


[~,pred] = max(predvalues, [], 2);
figure;
acc = sum(pred(:) == gt(:))/numel(gt);
 plotCM(gt ,pred, classes.name{2});
 title(layerNames{l});
 title(['camera parameter prediction, accuracy:' num2str(acc)], 'fontsize', 20);
 set (gcf, 'Units', 'normalized', 'Position', [0,0,1,1]);
 export_fig(gcf,  'env-pred.png', '-m1');





