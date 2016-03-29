% compute between class distance matrix

resSaveDir = '/lab/jiaping/papers/ECCV2016/results';

eval_deCNN = '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f18/iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000/maxActivation/test-evalInfo.mat';
eval_alexNet = '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f18/iLab20M-iLab_arc_de_dagnn_2streams_alexnet/maxActivation/test-evalInfo.mat';

nObjects = 10;
gap = 10;
nIntermediateLayers = 1;
%% dotproduct
distsaveName = 'iLab20M-deCNN-dotproduct.mat';
% deCNN
obj2obj_dist = cell(1, nIntermediateLayers);
rng('shuffle');
 

 	imdb_eval = load(eval_deCNN);
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
%                distmn = valuem' * valuen;     
%                
%                distmn = 1 - distmn./sqrt(diag(valuem'*valuem)*(diag(valuen'*valuen))');
%                
                distmn = dist2(valuem', valuen');
                distmn(distmn<0) = 0;
                distmn = sqrt(distmn);

%                 if m~= n
                    distl(m,n) = mean(distmn(:));
%                 else
%                     tmpidx = find(~tril(ones(size(distmn))));
%                     u = distmn(tmpidx);
%                     distl(m,n) = median(u);
%                 end
             end
         end         
         obj2obj_dist{l} = distl;
    end
    
    t = distl + distl' - diag(diag(distl));
    t = (t-min(t(:)))/(max(t(:)) - min(t(:)));
    figure; imagesc(t);    colormap(gray);

%     save(fullfile(resSaveDir, distsaveName), 'obj2obj_dist');
 

% alexnet
distsaveName = 'iLab20M-alexnet-dotproduct.mat';

obj2obj_dist = cell(1, nIntermediateLayers);

	imdb_eval = load(eval_alexNet);
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
%                distmn = 1 - distmn./sqrt(diag(valuem'*valuem)*(diag(valuen'*valuen))');

                distmn = dist2(valuem', valuen');
                distmn(distmn<0) = 0;
                distmn = sqrt(distmn);
%                 if m~= n
                    distl(m,n) = mean(distmn(:));
%                 else
%                     tmpidx = find(~tril(ones(size(distmn))));
%                     u = distmn(tmpidx);
%                     distl(m,n) = median(u);
%                 end
             end
         end         
         obj2obj_dist{l} = distl;
    end
    
       t = distl + distl' - diag(diag(distl));
    t = (t-min(t(:)))/(max(t(:)) - min(t(:)));
    figure; imagesc(t);    colormap(gray);

    
%     save(fullfile(resSaveDir, distsaveName), 'obj2obj_dist');
    
 