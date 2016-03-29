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

startIdx = 5;
gap     = 200;
nPoses  = 88;

tSNEsaveName =  'tSNE-fc2-2D-correct5.mat';

% w2cnn_MI
for a = 1:1

    if exist(fullfile(w2cnn_MI, maxsumFolderNames{a}, tSNEsaveName), 'file')
        continue;
    end
    
    evalFile = fullfile(w2cnn_MI, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    % t-SNE dimension reduction
    fc2values      =   evals.intermediateLayers(nIntermediateLayers).value;
    nImgs = size(fc2values,2);
    
    % biased
    sparseIdx = startIdx:gap:nImgs; 
    % correct way to get the sparse matrix
%     poselabel = evals.gt(2,:);
%     sparseIdx = [];
%     for p=1:nPoses
%         tmp = find(poselabel == p);
%         tmpsparse = 1:gap:numel(tmp);
%         tmp = tmp(tmpsparse);
%         sparseIdx = cat(2, sparseIdx, tmp);
%     end
    
     
    fc2values = fc2values(:,sparseIdx);
    
    
    labels = evals.gt(1,:);          
    labels = labels(sparseIdx);
    [fc22D, landmarks] = visualizeDescriptorsTSNE(fc2values', labels);            
%     fc22D = fc22D(landmarks,:);

    tSNEfc2.fc22D  = fc22D;
    tSNEfc2.fc2    = fc2values(:, landmarks);
    tmp_gt         = evals.gt(:,sparseIdx);
    tSNEfc2.gt     = tmp_gt(:, landmarks);
    tmp_pred       = evals.pred(:,sparseIdx);
    tSNEfc2.pred   = tmp_pred(:, landmarks);
    tmp_subsets     = evals.subset(sparseIdx);
    tSNEfc2.subset = tmp_subsets(landmarks);

    save(fullfile(w2cnn_MI, maxsumFolderNames{a}, tSNEsaveName), '-struct', 'tSNEfc2' , '-v7.3');
    clear tSNEfc2;
 
end
    

% w2cnn_I
for a = 1:0

    if exist(fullfile(w2cnn_I, maxsumFolderNames{a}, tSNEsaveName), 'file')
        continue;
    end 
    
    evalFile = fullfile(w2cnn_I, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    % t-SNE dimension reduction
    fc2values      =   evals.intermediateLayers(nIntermediateLayers).value;
    nImgs = size(fc2values,2);
    
    % biased
    sparseIdx = 1:gap:nImgs; 
    % correct way to get the sparse matrix
    poselabel = evals.gt(2,:);
    sparseIdx = [];
    for p=1:nPoses
        tmp = find(poselabel == p);
        tmpsparse = 1:gap:numel(tmp);
        tmp = tmp(tmpsparse);
        sparseIdx = cat(2, sparseIdx, tmp);
    end
    
    fc2values = fc2values(:,sparseIdx);
    
    
    labels = evals.gt(1,:);          
    labels = labels(sparseIdx);
    [fc22D, landmarks] = visualizeDescriptorsTSNE(fc2values', labels);            
%     fc22D = fc22D(landmarks,:);

    tSNEfc2.fc22D  = fc22D;
    tSNEfc2.fc2    = fc2values(:, landmarks);
    tmp_gt         = evals.gt(:,sparseIdx);
    tSNEfc2.gt     = tmp_gt(:, landmarks);
    tmp_pred       = evals.pred(:,sparseIdx);
    tSNEfc2.pred   = tmp_pred(:, landmarks);
    tmp_subsets     = evals.subset(sparseIdx);
    tSNEfc2.subset = tmp_subsets(landmarks);

    save(fullfile(w2cnn_I, maxsumFolderNames{a}, tSNEsaveName), '-struct', 'tSNEfc2' , '-v7.3');
    clear tSNEfc2;
 
end



 alexnetSparseIdx = sparseIdx;   
% alexnet
for a = 1:1

    if exist(fullfile(alexDir, maxsumFolderNames{a}, tSNEsaveName), 'file')
        continue;
    end
    
    evalFile = fullfile(alexDir, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    % t-SNE dimension reduction
    fc2values      =   evals.intermediateLayers(nIntermediateLayers).value;
    nImgs = size(fc2values,2);
    
    % biased
    sparseIdx = startIdx:gap:nImgs; 
%     % correct way to get the sparse matrix
%     poselabel = evals.gt(1,:);
%     sparseIdx = [];
%     for p=1:nPoses
%         tmp = find(poselabel == p);
%         tmpsparse = 1:gap:numel(tmp);
%         tmp = tmp(tmpsparse);
%         sparseIdx = cat(2, sparseIdx, tmp);
%     end    
%     sparseIdx = alexnetSparseIdx;
    
    
    fc2values = fc2values(:,sparseIdx);
    
    
    labels = evals.gt(1,:);          
    labels = labels(sparseIdx);
    [fc22D, landmarks] = visualizeDescriptorsTSNE(fc2values', labels);            
%     fc22D = fc22D(landmarks,:);

	tSNEfc2.fc22D  = fc22D;
    tSNEfc2.fc2    = fc2values(:, landmarks);
    tmp_gt         = evals.gt(:,sparseIdx);
    tSNEfc2.gt     = tmp_gt(:, landmarks);
    tmp_pred       = evals.pred(:,sparseIdx);
    tSNEfc2.pred   = tmp_pred(:, landmarks);
    tmp_subsets     = evals.subset(sparseIdx);
    tSNEfc2.subset = tmp_subsets(landmarks);

    save(fullfile(alexDir, maxsumFolderNames{a}, tSNEsaveName), '-struct', 'tSNEfc2' , '-v7.3');
    clear tSNEfc2;
 
end

