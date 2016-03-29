function infoPort = iLab_getBatchDagNN(imdb, batch, getBatch_opts, useGpu, ...
                                            infoPortVars, whoseLabels)
%=========================================================================
    getBatch_opts = iLab_nn_validateGetImageBatchParam(getBatch_opts);    
    images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
    im = iLab_getImageBatch(images, getBatch_opts) ;
%     im = cnn_imagenet_get_batch(images, getBatch_opts);
    if nargout > 0
        if useGpu
        	im = gpuArray(im) ;
        end

        if ischar(whoseLabels)
            whoseLabels = {whoseLabels};
        end
        
%         assert(numel(infoPortVars) == (1+numel(whoseLabels)));
        
        numAugments = getBatch_opts.numAugments;
        label_obj = imdb.images.label(1,batch);
        label_obj = repmat(label_obj, [numAugments 1]); 
        label_env = imdb.images.label(2,batch);
        label_env = repmat(label_env, [numAugments 1]);
        infoPort = {infoPortVars{1}, im};
        
        if numel(infoPortVars) == 1
            return;
        end
        
        if numel(whoseLabels) == 1
            switch whoseLabels{1}
                case 'object'
                    infoPort = cat(2, infoPort, {infoPortVars{2}, label_obj(:)});
                case 'environment'
                    infoPort = cat(2, infoPort, {infoPortVars{2}, label_env(:)});
                otherwise
                    error('un-recognized labels\n');
            end
        elseif numel(whoseLabels) == 2
            if ~ismember(whoseLabels{1}, {'object', 'environment'}) || ...
                    ~ismember(whoseLabels{2}, {'object', 'environment'})
                error('un-recognized labels\n');
            end
            infoPort = cat(2, infoPort, {infoPortVars{2}, label_obj(:)});
            infoPort = cat(2, infoPort, {infoPortVars{3}, label_env(:)});
        else
            error('Please specify whose label to be used\n');
        end 
        
    end

end