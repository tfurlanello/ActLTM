function fn = iLab_dagnn_evalBatch(whichLayersToEval, saveOpt)
% -------------------------------------------------------------------------
% basically, this function calls iLab_dagnn_predictBatch()

fn = @(net, imdb, subset, getBatch, opts_runTest) iLab_dagnn_predictBatch(net, imdb, subset, ...
                                                            getBatch, opts_runTest, ...
                                                            whichLayersToEval, saveOpt);

end