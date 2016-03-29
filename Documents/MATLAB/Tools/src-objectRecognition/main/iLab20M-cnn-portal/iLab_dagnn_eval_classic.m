% evaluate a classic deep architecture
%% note: we can only run dagnn model
 
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                       image database file
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
imdb_file = '/home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-paired/imdb.mat';
imdb = load(imdb_file);

saveDir = '/home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-paired/simplenn-visualization';


%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%%                              a classic linear-chain dagnn model
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
net_linear_file = '/home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-paired/simple/net-epoch-19.mat';
net_linear = load(net_linear_file);
net_linear = net_linear.net;
net_linear = dagnn.DagNN.loadobj(net_linear);
net_linear.conserveMemory = false;
net_linear.accumulateParamDers = false;
subsetTrain = find(imdb.images.set == 1);
subsetTest  = find(imdb.images.set == 3);
% subset      = [subsetTrain(:); subsetTest(:)];
subset = subsetTest;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- first save intermediate evaluation values 
whoseLabels         =   {'object'};
whichLayersToEval   =   {'pool1out', 'pool2out', 'relu3out', 'relu4out', 'pool5out', 'dropout7out'};

% max activation
saveOpt             =   'max';   
evalFileName = 'imdb-eval.mat';   
maxSaveDir = fullfile(saveDir, [saveOpt 'Activation']);
if ~exist(maxSaveDir, 'dir')
    mkdir(maxSaveDir);
end
maxEvalFile = fullfile(maxSaveDir, evalFileName);
if ~exist(maxEvalFile, 'file')
    [~, imdb_simple] =  iLab_dagnn_evalBatchPortal(net_linear, imdb, subset, whoseLabels, ...
                                                    whichLayersToEval, saveOpt);   
    save(maxEvalFile, '-struct', 'imdb_simple', '-v7.3');
    clear imdb_simple;
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
    [~, imdb_simple] =  iLab_dagnn_evalBatchPortal(net_linear, imdb, subset, whoseLabels, ...
                                                whichLayersToEval, saveOpt);   
    save(sumEvalFile, '-struct', 'imdb_simple', '-v7.3');
    clear imdb_simple;
end

evalFiles = {maxEvalFile, sumEvalFile};
evalDirs = {maxSaveDir, sumSaveDir};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- get receptive fields of designated layers

for f=numel(evalFiles):-1:1

    imdb_simple = load(evalFiles{f});
    simplennEval = imdb_simple.eval;
    clear imdb_simple;

    rf_saveDir = evalDirs{f};

    nStimuli = 100;
    nIntermediateLayers = numel(whichLayersToEval);
    assert( numel(simplennEval.intermediateLayers) == nIntermediateLayers);

    for l=1:(nIntermediateLayers-1)
        layerName   =   simplennEval.intermediateLayers(l).name;
        values      =   simplennEval.intermediateLayers(l).value;
        subset      =   simplennEval.subset;
        nchannels   =   size(values,1);

        sub_simplenn_saveDir = fullfile(rf_saveDir, layerName);
        if ~exist(sub_simplenn_saveDir, 'dir')
            mkdir(sub_simplenn_saveDir);
        end
        for c=1:nchannels

            fprintf(1, 'processing: layer-%s, channel-%d\n', layerName, c);
            
            saveName = strcat(layerName, '-c', num2str(c));
            if exist(fullfile(sub_simplenn_saveDir,  [saveName '-im.png']), 'file')
                continue;
            end

            c_values = values(c,:);
            [~, ind] = sort(c_values, 'descend');
            c_subset = subset(ind(1:nStimuli));
            whichLayerWhichChannel = struct('layer', layerName, ...
                                            'channel', c); 

            imdb_rf = iLab_dagnn_getImgCropFromReceptiveField(net_linear, imdb, c_subset, ...
                                                        whichLayerWhichChannel, whoseLabels);
%             save(fullfile(sub_simplenn_saveDir, [saveName '.mat']), '-struct', 'imdb_rf');

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

%     fc2values      =   simplennEval.intermediateLayers(nIntermediateLayers).value;
%     labels = simplennEval.gt;          
%     [fc22D, landmarks] = visualizeDescriptorsTSNE(fc2values', labels);            
%     fc22D = fc22D(landmarks,:);
% %     simplennEval        =   rmfield(simplennEval, {'intermediateLayers'}) ;
%     simplennEvalfc.fc22D  =   fc22D;
%     simplennEvalfc.fc2    =   fc2values;
%     simplennEvalfc.gt     = simplennEval.gt;
%     simplennEvalfc.pred   = simplennEval.pred;
%     simplennEvalfc.subset = simplennEval.subset;
%     
%     save(fullfile(sub_simplenn_saveDir, 'tSNE-fc2-2D.mat'), '-struct', 'simplennEvalfc' , '-v7.3');
%     clear simplennEvalfc simplennEval;
%     close all;
    clear simplennEval;
    
    %}

end

clear all;