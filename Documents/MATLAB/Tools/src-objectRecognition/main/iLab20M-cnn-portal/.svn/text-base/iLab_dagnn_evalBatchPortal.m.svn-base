function varargout = iLab_dagnn_evalBatchPortal(net, imdb, subset, whoseLabels, ...
                                                whichLayersToEval, saveOpt)
% running testing in a batch mode
% inputs
%        - net: trained network
%        - imdb: image datebase information
%        - modeltype: simplenn, dagnn
%        - inputinfo: {input, label_obj/label_env} (order can be shuffled)
%        - predictionNames: {prediction_obj, prediction_env} (you can't shuffle the order!!!)

%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                   parameters to get batch
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 

    dagnn_opts_getbatch.numThreads      =   12;
    dagnn_opts_getbatch.useGpu          =   true;
    dagnn_opts_getbatch.transformation  =   'stretch' ;
    dagnn_opts_getbatch.numAugments     =   1;
    dagnn_opts_getbatch = iLab_nn_validateGetImageBatchParam(dagnn_opts_getbatch);


%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                    parameters to run test
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    nn_opts_runBatch.batchSize       = 128 ;
    nn_opts_runBatch.numSubBatches   = 1 ;
    nn_opts_runBatch.prefetch        = false ;
    nn_opts_runBatch.gpus            = 1 ;
    nn_opts_runBatch.sync            = false ;
    nn_opts_runBatch.cudnn           = true ;
    nn_opts_runBatch.conserveMemory  = false ;
    nn_opts_runBatch = iLab_nn_validateRunTestParam(nn_opts_runBatch); 
    
    
    if ~exist('subset', 'var') || isempty(subset)
        subset = find(imdb.images.set == 3);
    end
    
    if ~exist('saveOpt', 'var') || isempty(saveOpt)
        saveOpt = 'max'; % 3 options: sum, max, raw
    end
    
 
    net.accumulateParamDers = 1;
    net.conserveMemory = 0;
    useGpu = 1;

    opts_getbatch   = vl_argparse(dagnn_opts_getbatch, net.meta.normalization);
    getBatch        = iLab_getBatchDagNNWrapper(opts_getbatch, useGpu, net.inputsNames, ...
                                                    whoseLabels);
%             subset          =  find(imdb.images.set==3);

    eval_fn = iLab_dagnn_evalBatch(whichLayersToEval, saveOpt);
    [labels, imdb] = eval_fn(net, imdb, subset, getBatch, nn_opts_runBatch);
    varargout = {labels, imdb};
 
        
end