function fn = iLab_de_getBatchDagNNWrapper(getBatch_opts, useGpu, infoPortVars, networkInputs)

    getBatch_opts = iLab_arg2struct(getBatch_opts);
    getBatch_opts = iLab_nn_validateGetImageBatchParam(getBatch_opts);
    fn = @(imdb,batch) iLab_de_getBatchDagNN(imdb,batch, ...
                                    getBatch_opts, useGpu, infoPortVars, networkInputs) ;

end