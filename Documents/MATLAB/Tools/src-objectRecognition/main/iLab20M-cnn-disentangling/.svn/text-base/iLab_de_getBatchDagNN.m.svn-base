function infoPort = iLab_de_getBatchDagNN(imdb, batch, getBatch_opts, useGpu, ...
                                            infoPortVars, networkInputs)
%=========================================================================
    inputsInfo = struct('imLeft', {}, ...
                        'imRight', {}, ...
                        'identity', {}, ...
                        'transformation', {});
%     networkInputs = {'imLeft', 'imRight', 'identity', 'transformation'};
%% notes
%   'infoPortVars' should have the same length as 'networkInputs', and
%   additionally, they should be corresponding to each other

    assert(numel(infoPortVars) == numel(networkInputs));

    getBatch_opts = iLab_nn_validateGetImageBatchParam(getBatch_opts);    
    imagesLeft  = strcat([imdb.imageDir filesep], imdb.images.name(1,batch)) ;
    if ismember('imLeft', networkInputs)
        imLeft = iLab_getImageBatch(imagesLeft, getBatch_opts);
    else
        imLeft = [];
    end
    imagesRight = strcat([imdb.imageDir filesep], imdb.images.name(2,batch)) ;
    
    if ismember('imRight', networkInputs)
        imRight = iLab_getImageBatch(imagesRight, getBatch_opts);
    else
        imRight = [];
    end
    
    if nargout > 0
        infoPort = {};
        cnt = 0;

        if useGpu
        	imLeft = gpuArray(imLeft) ;
            imRight = gpuArray(imRight);
        end
        
        %% there are two inputs & two outputs
        % left iamge, right image, object label, transformation label
%          assert(numel(infoPortVars) == 4);
        
        numAugments = getBatch_opts.numAugments;
        label_obj = imdb.images.label(1,batch);
        label_obj = repmat(label_obj, [numAugments 1]); 
        label_transform = imdb.images.label(2,batch);
        label_transform = repmat(label_transform, [numAugments 1]);
        
%         infoPort = {infoPortVars{1}, imLeft, infoPortVars{2}, imRight};

        if ~isempty(imLeft)
            cnt  = cnt + 1;
            infoPort = cat(2, infoPort, {infoPortVars{cnt}, imLeft});
        end
        
        if ~isempty(imRight)
            cnt = cnt + 1;
            infoPort = cat(2, infoPort, {infoPortVars{cnt}, imRight});
        end

        if ismember('identity', networkInputs)
            cnt = cnt + 1;
            infoPort = cat(2, infoPort, {infoPortVars{cnt}, label_obj(:)});
        end
        
        if ismember('transformation', networkInputs)
            cnt = cnt + 1;
            infoPort = cat(2, infoPort, {infoPortVars{cnt}, label_transform(:)});
        end
        
    end

end