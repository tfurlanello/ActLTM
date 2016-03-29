% evaluate a deep architecture with multi-layer injection
%% note: we can only run dagnn model
 
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
%%                                            a trained dagnn model
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
net_file = '/home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-1024/net-epoch-10.mat';
net = load(net_file);
net = net.net;
net = dagnn.DagNN.loadobj(net);
net.conserveMemory = false;
net.accumulateParamDers = false;
subsetTrain = find(imdb.images.set == 1);
subsetTest  = find(imdb.images.set == 3);
% subset      = [subsetTrain(:); subsetTest(:)];
subset = subsetTest;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- first save intermediate evaluation values 
whoseLabels         =   {'object', 'environment'};
whichLayersToEval   =   {'pool1out', 'pool2out', 'relu3out', 'relu4out', 'pool5out', 'dropout7out', ...
                             'conv32dropoutout', 'conv42dropoutout'};

% max activation
saveOpt             =   'max';   
evalFileName        = 'imdb-eval.mat';   
maxSaveDir = fullfile(saveDir, [saveOpt 'Activation']);
if ~exist(maxSaveDir, 'dir')
    mkdir(maxSaveDir);
end
maxEvalFile = fullfile(maxSaveDir, evalFileName);
if ~exist(maxEvalFile, 'file')
    [~, imdb_complex] =  iLab_dagnn_evalBatchPortal(net, imdb, subset, whoseLabels, ...
                                                    whichLayersToEval, saveOpt);   
    save(maxEvalFile, '-struct', 'imdb_complex', '-v7.3');
    clear imdb_complex;
end

% sum activation
saveOpt             =   'sum';     
evalFileName = 'imdb-eval.mat';   
sumSaveDir = fullfile(saveDir, [saveOpt 'Activation']);
if ~exist(sumSaveDir, 'dir')
    mkdir(sumSaveDir);
end
sumEvalFile = fullfile(sumSaveDir, evalFileName);
if ~exist(sumEvalFile, 'file')
    [~, imdb_complex] =  iLab_dagnn_evalBatchPortal(net, imdb, subset, whoseLabels, ...
                                                whichLayersToEval, saveOpt);   
    save(sumEvalFile, '-struct', 'imdb_complex', '-v7.3');
    clear imdb_complex;
end

evalFiles = {maxEvalFile, sumEvalFile};
evalDirs = {maxSaveDir, sumSaveDir};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- get receptive fields of designated layers

for f=1:numel(evalFiles)

    imdb_complex = load(evalFiles{f});
    complexEval = imdb_complex.eval;
    clear imdb_complex;

    rf_saveDir = evalDirs{f};

    nStimuli = 81;
    nIntermediateLayers = numel(whichLayersToEval);
    assert( numel(complexEval.intermediateLayers) == nIntermediateLayers);

     
    for l=1:(nIntermediateLayers-3)
        layerName   =   complexEval.intermediateLayers(l).name;
        values      =   complexEval.intermediateLayers(l).value;
        subset      =   complexEval.subset;
        nchannels   =   size(values,1);

        sub_complex_saveDir = fullfile(rf_saveDir, layerName);
        if ~exist(sub_complex_saveDir, 'dir')
            mkdir(sub_complex_saveDir);
        end
        for c=1:nchannels

            fprintf(1, 'processing: layer-%s, channel-%d\n', layerName, c);
            
            saveName = strcat(layerName, '-c', num2str(c));
            if exist(fullfile(sub_complex_saveDir, [saveName '-im.png']), 'file')
                continue;
            end

            c_values = values(c,:);
            [~, ind] = sort(c_values, 'descend');
            c_subset = subset(ind(1:nStimuli));
            whichLayerWhichChannel = struct('layer', layerName, ...
                                            'channel', c); 

            imdb_rf = iLab_dagnn_getImgCropFromReceptiveField(net, imdb, c_subset, ...
                                                        whichLayerWhichChannel, whoseLabels);
%             save(fullfile(sub_complex_saveDir, [saveName '.mat']), '-struct', 'imdb_rf');

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
            imwrite(crops,   fullfile(sub_complex_saveDir, [saveName '-rf.png']), 'png');

        end
    end
     
    % t-SNE dimension reduction
% 	fc2values      =   complexEval.intermediateLayers(nIntermediateLayers-3).value;
%     labels = complexEval.gt;          
%     [fc22D, landmarks] = visualizeDescriptorsTSNE(fc2values', labels);            
%     fc22D = fc22D(landmarks,:);
% %     complexEval =   rmfield(complexEval, {'intermediateLayers'}) ;
%     complexEvalfc.fc22D  =   fc22D;
%     complexEvalfc.fc2    =   fc2values;
%     complexEvalfc.gt     = complexEval.gt;
%     complexEvalfc.pred   = complexEval.pred;
%     complexEvalfc.subset = complexEval.subset;
%     
%     save(fullfile(sub_complex_saveDir, 'tSNE-fc2-2D.mat'), '-struct', 'complexEvalfc' , '-v7.3');
%     clear complexEvalfc;
%     close all;
    
    % environmental label prediction
    tmp = [41 34 25];
    for l=1:3
        whichLayer = whichLayersToEval{end-l+1};
        values      =   complexEval.intermediateLayers(nIntermediateLayers-l+1).value;
        params = net.layers(tmp(l)).params;
        
        filt = squeeze(net.params(net.getParamIndex(params{1})).value);
        bias = net.params(net.getParamIndex(params{2})).value;
        
        predout = values'*filt + repmat((bias(:))', [size(values,2) 1]); 

        complexEvalfcE.(whichLayer)    =   predout;
        complexEvalfcE.gt     = complexEval.gt;
        complexEvalfcE.pred   = complexEval.pred;
        complexEvalfcE.subset = complexEval.subset;
        
        save(fullfile(rf_saveDir, [whichLayer '-pred-evn.mat']), '-struct', 'complexEvalfcE' , '-v7.3');
        clear complexEvalfcE;

        
    end
    
    clear complexEval;
    
    
    %}

end

clear all;