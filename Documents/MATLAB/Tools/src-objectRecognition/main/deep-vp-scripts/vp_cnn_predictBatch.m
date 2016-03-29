function varargout = vp_cnn_predictBatch(net, imdb, subset, modeltype,...
                                                    whoseLabels, whichLayersToEval)
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
    simplenn_opts_getbatch.numThreads       =   12;
    simplenn_opts_getbatch.transformation   =   'stretch' ;
    simplenn_opts_getbatch.numAugments      =   1;
    simplenn_opts_getbatch = iLab_nn_validateGetImageBatchParam(simplenn_opts_getbatch);

    dagnn_opts_getbatch.numThreads      =   12;
    dagnn_opts_getbatch.useGpu          =   true;
    dagnn_opts_getbatch.transformation  =   'stretch' ;
    dagnn_opts_getbatch.numAugments     =   1;
    dagnn_opts_getbatch = iLab_nn_validateGetImageBatchParam(dagnn_opts_getbatch);


%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                    parameters to run test
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    nn_opts_runBatch.batchSize       = 128 ; %% changed batch size
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
    
    switch modeltype
        case 'simplenn'
            opts_getbatch   = vl_argparse(simplenn_opts_getbatch, net.normalization) ;
            getBatch        = vp_getBatchSimpleNNWrapper(opts_getbatch, whoseLabel);
%             subset          =  find(imdb.images.set==3);

            labels = iLab_simplenn_predictBatch(net, imdb, subset, getBatch, nn_opts_runBatch);
            varargout = {labels};
            return;
            
        case 'dagnn'
            net.accumulateParamDers = 1;
            net.conserveMemory = 0;
            useGpu = 1;

            opts_getbatch   = vl_argparse(dagnn_opts_getbatch, net.meta.normalization);
            getBatch        = vp_getBatchDagNNWrapper(opts_getbatch, useGpu, net.inputsNames, ...
                                                            whoseLabels);
%             subset          =  find(imdb.images.set==3);

            [labels, imdb] = iLab_dagnn_predictBatch(net, imdb, subset, getBatch, ...
                                                        nn_opts_runBatch, whichLayersToEval);
            varargout = {labels, imdb};
            return;
        otherwise
            error('un-recognized model type\n');
    end
        
end