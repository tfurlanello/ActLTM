function infoPort = vp_getBatchDagNN_MTL(imdb, batch, getBatch_opts, useGpu, ...
                                            infoPortVars, label2idx)
%=========================================================================
    getBatch_opts = iLab_nn_validateGetImageBatchParam(getBatch_opts);    
    images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
    im = vp_getImageBatch(images, getBatch_opts) ;
    if nargout > 0
        if useGpu
        	im = gpuArray(im) ;
        end 
        
        nLabels = numel(keys(label2idx));
        assert(numel(infoPortVars) == (1+nLabels));
        
        numAugments = getBatch_opts.numAugments;
        infoPort = {infoPortVars{1}, im};
        
        
        for l=1:nLabels            
            labelName = infoPortVars{l+1};
            labelValues = imdb.images.label(l,batch);
            labelValues = repmat(labelValues, [numAugments 1]);
            infoPort = cat(2, infoPort, {labelName, labelValues(:)});            
        end
        
        
    end

end