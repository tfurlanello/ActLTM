%% use t-SNE to visualize top-layer features


saveDir = '';

maxsumFolderNames =  {'maxActivation', 'sumActivation'};
evalFileName      =  'imdb-eval.mat';
nActivationTypes = numel(maxsumFolderNames);
%% (1) compute correlation of pose and identity entropy

alexDir = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-obj/visualization';
w2cnn_I = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-fc2/visualization'; 
w2cnn_MI = '/home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/visualization'; 


whichLayersToEval   =  {'pool1out', 'pool2out', 'relu3out', 'relu4out', ...
                                 'pool5out', 'dropout6out', 'dropout7out'};
layerNames          =  {'pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7'};                                 
nIntermediateLayers =  numel(whichLayersToEval);                                 
nEvals              =  numel(whichLayersToEval);     

gap = 20;
% w2cnn_MI
for a = 1:nActivationTypes

    evalFile = fullfile(w2cnn_MI, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    % t-SNE dimension reduction
    fc2values      =   evals.intermediateLayers(nIntermediateLayers).value;
    nImgs = size(fc2values,2);
    sparseIdx = 1:gap:nImgs; 
    fc2values = fc2values(:,sparseIdx);
    
    
    labels = evals.gt(1,:);          
    labels = labels(sparseIdx);
    [fc22D, landmarks] = visualizeDescriptorsTSNE(fc2values', labels);            
    fc22D = fc22D(landmarks,:);

    tSNEfc2.fc22D  = fc22D;
    tSNEfc2.fc2    = fc2values;
    tSNEfc2.gt     = evals.gt(:,sparseIdx);
    tSNEfc2.pred   = evals.pred(:,sparseIdx);
    tSNEfc2.subset = evals.subset(sparseIdx);

    save(fullfile(w2cnn_MI, maxsumFolderNames{a}, 'tSNE-fc2-2D.mat'), '-struct', 'tSNEfc2' , '-v7.3');
    clear tSNEfc2;
 
end
    
    
% alexnet
for a = 1:nActivationTypes

    evalFile = fullfile(alexDir, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    % t-SNE dimension reduction
    fc2values      =   evals.intermediateLayers(nIntermediateLayers).value;
    nImgs = size(fc2values,2);
    sparseIdx = 1:gap:nImgs;
    fc2values = fc2values(:,sparseIdx);
    
    
    labels = evals.gt(1,:);          
    labels = labels(sparseIdx);
    [fc22D, landmarks] = visualizeDescriptorsTSNE(fc2values', labels);            
    fc22D = fc22D(landmarks,:);

    tSNEfc2.fc22D  = fc22D;
    tSNEfc2.fc2    = fc2values;
    tSNEfc2.gt     = evals.gt(:,sparseIdx);
    tSNEfc2.pred   = evals.pred(:,sparseIdx);
    tSNEfc2.subset = evals.subset(sparseIdx);

    save(fullfile(alexDir, maxsumFolderNames{a}, 'tSNE-fc2-2D.mat'), '-struct', 'tSNEfc2' , '-v7.3');
    clear tSNEfc2;
 
end



% w2cnn_I
for a = 1:nActivationTypes

    evalFile = fullfile(w2cnn_I, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    % t-SNE dimension reduction
    fc2values      =   evals.intermediateLayers(nIntermediateLayers).value;
    nImgs = size(fc2values,2);
    sparseIdx = 1:gap:nImgs;
    fc2values = fc2values(:,sparseIdx);
    
    
    labels = evals.gt(1,:);          
    labels = labels(sparseIdx);
    [fc22D, landmarks] = visualizeDescriptorsTSNE(fc2values', labels);            
    fc22D = fc22D(landmarks,:);

    tSNEfc2.fc22D  = fc22D;
    tSNEfc2.fc2    = fc2values;
    tSNEfc2.gt     = evals.gt(:,sparseIdx);
    tSNEfc2.pred   = evals.pred(:,sparseIdx);
    tSNEfc2.subset = evals.subset(sparseIdx);

    save(fullfile(w2cnn_I, maxsumFolderNames{a}, 'tSNE-fc2-2D.mat'), '-struct', 'tSNEfc2' , '-v7.3');
    clear tSNEfc2;
 
end

