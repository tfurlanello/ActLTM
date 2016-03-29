% compute mean AP


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

nObjects = 10;


% w2cnn_MI

for a = 1:nActivationTypes

    acc_obj = zeros(1, nObjects);
    evalFile = fullfile(w2cnn_MI, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    % t-SNE dimension reduction
    gt    =  evals.gt(1,:);
    pred  =  evals.pred(1,:);
    
    for o=1:nObjects
        idx     =  (gt == o);
        o_pred  =  pred(idx);
        
        acc_obj(o) = sum(o_pred == o)/numel(o_pred);
        
    end
    mAP = acc_obj;
    save(fullfile(w2cnn_MI, maxsumFolderNames{a}, 'mAP.mat'), 'mAP');
    
end


% alex_net 

for a = 1:nActivationTypes

    acc_obj = zeros(1, nObjects);
    evalFile = fullfile(alexDir, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    % t-SNE dimension reduction
    gt    =  evals.gt(1,:);
    pred  =  evals.pred(1,:);
    
    for o=1:nObjects
        idx     =  (gt == o);
        o_pred  =  pred(idx);
        
        acc_obj(o) = sum(o_pred == o)/numel(o_pred);
        
    end
    mAP = acc_obj;
    save(fullfile(alexDir, maxsumFolderNames{a}, 'mAP.mat'), 'mAP');
    
end


% w2_cnn_I 
for a = 1:nActivationTypes

    acc_obj = zeros(1, nObjects);
    evalFile = fullfile(w2cnn_I, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    % t-SNE dimension reduction
    gt    =  evals.gt(1,:);
    pred  =  evals.pred(1,:);
    
    for o=1:nObjects
        idx     =  (gt == o);
        o_pred  =  pred(idx);
        
        acc_obj(o) = sum(o_pred == o)/numel(o_pred);
        
    end
    mAP = acc_obj;
    save(fullfile(w2cnn_I, maxsumFolderNames{a}, 'mAP.mat'), 'mAP');
    
end
