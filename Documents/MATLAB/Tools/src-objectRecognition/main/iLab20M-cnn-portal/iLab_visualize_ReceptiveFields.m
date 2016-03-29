% to visualize receptive fields of a trained dagnn model

%% note: we can only run dagnn model

 
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                       image database file
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
imdb_file = '/home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-paired/imdb.mat';
imdb = load(imdb_file);

saveDir = '/home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-paired';


%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%%                              a classic linear-chain dagnn model
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
net_linear_file = '/home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-paired/simple/net-epoch-19.mat';
net_linear = load(net_linear_file);
net_linear = net_linear.net;
net_linear = dagnn.DagNN.loadobj(net_linear);
subsetTrain = find(imdb.images.set == 1);
subsetTest  = find(imdb.images.set == 3);
% subset      = [subsetTrain(:); subsetTest(:)];
subset = subsetTest;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- first save intermediate evaluation values 
modeltype           =   'dagnn';
whoseLabels         =   {'object'};
whichLayersToEval   =   {'pool1out', 'pool2out', 'relu3out', 'relu4out', 'pool5out'};
[~, imdb_simple] = iLab_cnn_predictBatch(net_linear, imdb, subset, modeltype,...
                                                    whoseLabels, whichLayersToEval);
                                                
evalFileName = 'imdb-simple-eval.mat';                                                
save(fullfile(saveDir, evalFileName), '-struct', 'imdb_simple', '-v7.3');

simplennEval = imdb_simple.prediction;
clear imdb_simple;

simplenn_saveDir = strcat(saveDir, filesep, 'simplenn-visualization');
if ~exist(simplenn_saveDir, 'dir')
    mkdir(simplenn_saveDir);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- get receptive fields of designated layers
nStimuli = 100;
nIntermediateLayers = numel(whichLayersToEval);
assert( numel(simplennEval.intermediateLayers) == nIntermediateLayers);

for l=1:nIntermediateLayers
    layerName   =   simplennEval.intermediateLayers(l).name;
    values      =   simplennEval.intermediateLayers(l).value;
    subset      =   simplennEval.subset;
    nchannels   =   size(values,1);
    
    sub_simplenn_saveDir = fullfile(simplenn_saveDir, layerName);
    if ~exist(sub_simplenn_saveDir, 'dir')
        mkdir(sub_simplenn_saveDir);
    end
    for c=1:nchannels
        
        fprintf(1, 'processing: layer-%s, channel-%d\n', layerName, c);
        
        c_values = values(c,:);
        [~, ind] = sort(c_values, 'descend');
        c_subset = subset(ind(1:nStimuli));
        whichLayerWhichChannel = struct('layer', layerName, ...
                                        'channel', c); 
        
        imdb_rf = iLab_dagnn_getImgCropFromReceptiveField(net_linear, imdb, c_subset, ...
                                                    whichLayerWhichChannel, whoseLabels);
        saveName = strcat(layerName, '-c', num2str(c));
        save(fullfile(sub_simplenn_saveDir, [saveName '.mat']), '-struct', 'imdb_rf');
        
        rawImgs = [];
        crops = [];
        
        for s=1:numel(imdb_rf.stimuli.crop)
            tmpImg = imdb_rf.stimuli.crop(s).image;
            rawImgs = cat(2, rawImgs, tmpImg(:));
            tmpPatch = imdb_rf.stimuli.crop(s).patch;
            crops = cat(2, crops, tmpPatch(:));
        end
        
        szImg = round(sqrt(size(rawImgs,1)/3));
        szCrop = round(sqrt(size(crops,1)/3));
        
        rawImgs = uint8(rawImgs);
        crops = uint8(crops);
        
        rawImgs = imCollage(rawImgs, [szImg szImg]);
        crops = imCollage(crops, [szCrop szCrop]);
        
        imwrite(rawImgs, fullfile(sub_simplenn_saveDir, [saveName '-im.png']), 'png');
        imwrite(crops, fullfile(sub_simplenn_saveDir, [saveName '-rf.png']), 'png');
        
    end
end
%}

clear all;
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                       image database file
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
imdb_file = '/home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-1024/imdb.mat';
imdb = load(imdb_file);

saveDir = '/home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-1024/visualization';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end


%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%%                                     manually-crafted dagnn model
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
net_file = '/home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-1024/net-epoch-10.mat';
net = load(net_file);
net = net.net;
net = dagnn.DagNN.loadobj(net);
subsetTrain = find(imdb.images.set == 1);
subsetTest  = find(imdb.images.set == 3);
% subset      = [subsetTrain(:); subsetTest(:)];
subset = subsetTest;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- first save intermediate evaluation values 
modeltype           =   'dagnn';
whoseLabels         =   {'object', 'environment'};
whichLayersToEval   =   {'pool1out', 'pool2out', 'relu3out', 'relu4out', 'pool5out'};
[~, imdb_complex] = iLab_cnn_predictBatch(net, imdb, subset, modeltype,...
                                                    whoseLabels, whichLayersToEval);
                                                
evalFileName = 'imdb-complex-eval.mat';                                                
save(fullfile(saveDir, evalFileName), '-struct', 'imdb_complex', '-v7.3');

complexEval = imdb_complex.prediction;
clear imdb_complex;

complexnn_saveDir = saveDir;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- get receptive fields of designated layers
nStimuli = 100;
nIntermediateLayers = numel(whichLayersToEval);
assert( numel(complexEval.intermediateLayers) == nIntermediateLayers);

for l=1:nIntermediateLayers
    layerName   =   complexEval.intermediateLayers(l).name;
    values      =   complexEval.intermediateLayers(l).value;
    subset      =   complexEval.subset;
    nchannels   =   size(values,1);
    
    sub_complex_saveDir = fullfile(complexnn_saveDir, layerName);
    if ~exist(sub_complex_saveDir, 'dir')
        mkdir(sub_complex_saveDir);
    end
    for c=1:nchannels
        
        fprintf(1, 'processing: layer-%s, channel-%d\n', layerName, c);
        
        c_values = values(c,:);
        [~, ind] = sort(c_values, 'descend');
        c_subset = subset(ind(1:nStimuli));
        whichLayerWhichChannel = struct('layer', layerName, ...
                                        'channel', c); 
        
        imdb_rf = iLab_dagnn_getImgCropFromReceptiveField(net, imdb, c_subset, ...
                                                    whichLayerWhichChannel, whoseLabels);

        saveName = strcat(layerName, '-c', num2str(c));
        save(fullfile(sub_complex_saveDir, [saveName '.mat']), '-struct', 'imdb_rf');
        
        rawImgs = [];
        crops = [];
        
        for s=1:numel(imdb_rf.stimuli.crop)
            tmpImg = imdb_rf.stimuli.crop(s).image;
            rawImgs = cat(2, rawImgs, tmpImg(:));
            tmpPatch = imdb_rf.stimuli.crop(s).patch;
            crops = cat(2, crops, tmpPatch(:));
        end
        
        szImg = round(sqrt(size(rawImgs,1)/3));
        szCrop = round(sqrt(size(crops,1)/3));
        
        rawImgs = uint8(rawImgs);
        crops = uint8(crops);
        
        rawImgs = imCollage(rawImgs, [szImg szImg]);
        crops = imCollage(crops, [szCrop szCrop]);
        
        imwrite(rawImgs, fullfile(sub_complex_saveDir, [saveName '-im.png']), 'png');
        imwrite(crops, fullfile(sub_complex_saveDir, [saveName '-rf.png']), 'png');                                                
                                                
    end
end
                                                




