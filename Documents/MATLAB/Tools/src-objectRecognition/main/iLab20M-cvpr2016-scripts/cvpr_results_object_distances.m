% compute viewpoint dependent response

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
gap = 10;
%% dotproduct
distsaveName = 'dotproduct.mat';
% w2cnn_MI
obj2obj_dist = cell(1, nIntermediateLayers);
rng('shuffle');
for a = 1:1
    acc_obj = zeros(1, nObjects);
    evalFile = fullfile(w2cnn_MI, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    gt  = evals.gt(1,:);    
    
    for l=1:nIntermediateLayers
        
         lvalues = evals.intermediateLayers(l).value;
         distl = zeros(nObjects, nObjects);
         for m=1:nObjects
             idxm = find(gt == m);
             tmpidx = randperm(numel(idxm));
             idxm = idxm(tmpidx(1:round(numel(idxm)/gap)));
             valuem = single(lvalues(:, idxm));
             for n=m:nObjects
                 fprintf(1, 'Layer: %d, obj-obj: %d-%d\n', l, m,n);
                idxn = find(gt == n);
                tmpidx = randperm(numel(idxn));
                idxn = idxn(tmpidx(1:round(numel(idxn)/gap)));
                valuen = single(lvalues(:,idxn));                
                distmn = valuem' * valuen;                
%                 if m~= n
                    distl(m,n) = median(distmn(:));
%                 else
%                     tmpidx = find(~tril(ones(size(distmn))));
%                     u = distmn(tmpidx);
%                     distl(m,n) = median(u);
%                 end
             end
         end         
         obj2obj_dist{l} = distl;
    end
    
    save(fullfile(w2cnn_MI, maxsumFolderNames{a}, distsaveName), 'obj2obj_dist');
    
end


% alexnet
obj2obj_dist = cell(1, nIntermediateLayers);

for a = 1:1
    acc_obj = zeros(1, nObjects);
    evalFile = fullfile(alexDir, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    gt  = evals.gt(1,:);    
    
    for l=1:nIntermediateLayers
        
         lvalues = evals.intermediateLayers(l).value;
         distl = zeros(nObjects, nObjects);
         for m=1:nObjects
             idxm = find(gt == m);
             tmpidx = randperm(numel(idxm));
             idxm = idxm(tmpidx(1:round(numel(idxm)/gap)));
             valuem = single(lvalues(:, idxm));
             for n=m:nObjects
                 fprintf(1, 'Layer: %d, obj-obj: %d-%d\n', l, m,n);
                idxn = find(gt == n);
                tmpidx = randperm(numel(idxn));
                idxn = idxn(tmpidx(1:round(numel(idxn)/gap)));
                valuen = single(lvalues(:,idxn));                
                distmn = valuem' * valuen;                
%                 if m~= n
                    distl(m,n) = median(distmn(:));
%                 else
%                     tmpidx = find(~tril(ones(size(distmn))));
%                     u = distmn(tmpidx);
%                     distl(m,n) = median(u);
%                 end
             end
         end         
         obj2obj_dist{l} = distl;
    end
    
    save(fullfile(alexDir, maxsumFolderNames{a}, distsaveName), 'obj2obj_dist');
    
end


%% l2 distance
distsaveName = 'l2distance.mat';
% w2cnn_MI
obj2obj_dist = cell(1, nIntermediateLayers);

for a = 1:1
    acc_obj = zeros(1, nObjects);
    evalFile = fullfile(w2cnn_MI, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    gt  = evals.gt(1,:);    
    
    for l=1:nIntermediateLayers
        
         lvalues = evals.intermediateLayers(l).value;
         distl = zeros(nObjects, nObjects);
         for m=1:nObjects
             idxm = find(gt == m);
             tmpidx = randperm(numel(idxm));
             idxm = idxm(tmpidx(1:round(numel(idxm)/gap)));
             valuem = single(lvalues(:, idxm));
             for n=m:nObjects
                 fprintf(1, 'Layer: %d, obj-obj: %d-%d\n', l, m,n);
                idxn = find(gt == n);
                tmpidx = randperm(numel(idxn));
                idxn = idxn(tmpidx(1:round(numel(idxn)/gap)));
                valuen = single(lvalues(:,idxn));                
%                 distmn = valuem' * valuen;                
%                 if m~= n
                     distmn = sqrt(dist2(valuem', valuen'));
%                 else
%                     tmpidx = find(~tril(ones(size(distmn))));
%                     u = distmn(tmpidx);
%                     distl(m,n) = median(u);
%                 end
             end
         end         
         obj2obj_dist{l} = distl;
    end
    
    
    save(fullfile(w2cnn_MI, maxsumFolderNames{a}, distsaveName), 'obj2obj_dist');
    
end


% alexnet
obj2obj_dist = cell(1, nIntermediateLayers);

for a = 1:1
    acc_obj = zeros(1, nObjects);
    evalFile = fullfile(alexDir, maxsumFolderNames{a}, evalFileName);
    imdb_eval = load(evalFile);
    evals = imdb_eval.eval;
    clear imdb_eval;
    
    gt  = evals.gt(1,:);    
    
    for l=1:nIntermediateLayers
        
         lvalues = evals.intermediateLayers(l).value;
         distl = zeros(nObjects, nObjects);
         for m=1:nObjects
             idxm = find(gt == m);
             tmpidx = randperm(numel(idxm));
             idxm = idxm(tmpidx(1:round(numel(idxm)/gap)));
             valuem = single(lvalues(:, idxm));
             for n=m:nObjects
                 fprintf(1, 'Layer: %d, obj-obj: %d-%d\n', l, m,n);
                idxn = find(gt == n);
                tmpidx = randperm(numel(idxn));
                idxn = idxn(tmpidx(1:round(numel(idxn)/gap)));
                valuen = single(lvalues(:,idxn));                
%                 distmn = valuem' * valuen;                
%                 if m~= n
                     distmn = sqrt(dist2(valuem', valuen'));
%                 else
%                     tmpidx = find(~tril(ones(size(distmn))));
%                     u = distmn(tmpidx);
%                     distl(m,n) = median(u);
%                 end
             end
         end         
         obj2obj_dist{l} = distl;
    end
    
    save(fullfile(alexDir, maxsumFolderNames{a}, distsaveName), 'obj2obj_dist');
    
end








