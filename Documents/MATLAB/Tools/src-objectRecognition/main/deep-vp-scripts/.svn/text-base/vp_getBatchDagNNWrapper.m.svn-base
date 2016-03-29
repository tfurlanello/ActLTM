function fn = vp_getBatchDagNNWrapper(getBatch_opts, useGpu, infoPortVars, whoseLabels)

    getBatch_opts = iLab_arg2struct(getBatch_opts);
    getBatch_opts = iLab_nn_validateGetImageBatchParam(getBatch_opts);
    fn = @(imdb,batch) vp_getBatchDagNN(imdb,batch, getBatch_opts, useGpu, ...
                                                            infoPortVars, whoseLabels) ;

end