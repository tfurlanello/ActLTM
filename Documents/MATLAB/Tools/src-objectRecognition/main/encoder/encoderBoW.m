function [sig, modelBoW,  success] = encoderBoW(bof, K)

    narginchk(2,2);
    success = true;
    nSamples  =   numel(bof);
     
    if iscell(bof{1})
        bcell = true; bmat = false;
    elseif ismatrix(bof{1})
        bcell = false; bmat = true;
    else
        error('Format error for the 1st two parameters\n');
    end
    
    idxSamples = [];
    bofeats = [];
    for i=1:nSamples
        idxSamples     = cat(1, idxSamples, i*ones(size(bof{i},1),1));
        if bcell
            bofeats = cat(1, bofeats, cell2mat(bof{i}));
        elseif bmat
            bofeats = cat(1, bofeats, bof{i});
        end
    end
    
    if size(bofeats,1) < 3*K
        success = false;
        modelBoW = [];
        return;
    end
    
    bofeats = double(bofeats);
    fprintf(1,'K-means to get the vocabulary\n');
    [idx2CTrain,Centers] = ...
        kmeans(bofeats, K, 'emptyaction','singleton', 'replicates', 1);
    
        % bow of training samples
    idx2CTraining = cell(nSamples,1); 
    sig = zeros(nSamples,K);
    for i=1:nSamples
        flag = idxSamples == i;
        idx2CTraining{i} = idx2CTrain(flag);
        [sig(i,:), ~] = hist(idx2CTraining{i}, 1:K);
    end

    modelBoW = validateBoWmodel;
    modelBoW.clusterCenters = Centers;
    
end