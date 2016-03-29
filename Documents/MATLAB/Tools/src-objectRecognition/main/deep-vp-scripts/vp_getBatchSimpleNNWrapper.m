function fn = vp_getBatchSimpleNNWrapper(getBatch_opts, whoseLabel)
    % -------------------------------------------------------------------------
    getBatch_opts = iLab_nn_validateGetImageBatchParam(getBatch_opts);
    fn = @(imdb,batch) vp_getBatchSimpleNN(imdb,batch,getBatch_opts, whoseLabel) ;
    
end