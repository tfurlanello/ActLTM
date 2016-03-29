 


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



% w2cnn_MI

for a = 1:nActivationTypes

    mActivations = cell(1, nEvals);
    
    evalFile = fullfile(w2cnn_MI, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    for l=1:nIntermediateLayers        
        act =  evals.intermediateLayers(l).value;
        mActivations{l} = mean(act,2);
    end
        
    save(fullfile(w2cnn_MI, maxsumFolderNames{a}, 'mActivations.mat'), 'mActivations');
    
end


% alex_net 

for a = 1:nActivationTypes

    mActivations = cell(1, nEvals);
    
    evalFile = fullfile(alexDir, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    for l=1:nIntermediateLayers        
        act =  evals.intermediateLayers(l).value;
        mActivations{l} = mean(act,2);
    end
        
    save(fullfile(alexDir, maxsumFolderNames{a}, 'mActivations.mat'), 'mActivations');
    
end


% w2_cnn_I 
for a = 1:nActivationTypes

    mActivations = cell(1, nEvals);
    
    evalFile = fullfile(w2cnn_I, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    for l=1:nIntermediateLayers        
        act =  evals.intermediateLayers(l).value;
        mActivations{l} = mean(act,2);
    end
        
    save(fullfile(w2cnn_I, maxsumFolderNames{a}, 'mActivations.mat'), 'mActivations');
    
end



 