function varargout = iLab_dagnn_streams_alexnet(nstreams, bshare, args)  

%% functionality
% build several parallel streams, using classic alexnet
% in default, all parallel streams share the same filters
% but, we could build individual streams using different parameter settings

%% inputs
%       nstream     -  # of parallel streams
%       bshare      -  wheather to share parameters or nor (true in
%                      default)
%       args        -  it contains default settings for parameter
%                      intializations

%% note:
% the output of the architecture is the prediction
% don't add loss layer as the last layer; we will have much more
% flexibility to use different manually-crafted loss functions 

    narginchk(0,3);
    if ~exist('nstreams', 'var') || isempty(nstreams)
        nstreams = 1;
    end
    
    if ~exist('bshare', 'var') || isempty(bshare)
        bshare = true;
    end
    
    if ~exist('args', 'var') || isempty(args)
        args = {};
    end

    opts.batchNormalization = false ;
    
    opts.conv.weightInitMethod = 'gaussian';
    opts.conv.scale         = 1.0;
    opts.conv.learningRate  = [1 2];
    opts.conv.weightDecay   = [1 0];
	opts.fc.size            = 1024;

    
    opts.bnorm.learningRate = [2 1];
    opts.bnorm.weightDecay  = [0 0];
    
    opts.norm.param         = [5 1 0.0001/5 0.75];
    opts.pooling.method     = 'max';
    opts.dropout.rate       = 0.5;
    
    opts = vl_argparse(opts, args) ;
    
    LayerNames      =   cell(nstreams,1);
    LayerInputs     =   cell(nstreams,1);
    LayerOutputs    =   cell(nstreams,1);
	fcsize = opts.fc.size;

    % -------------------------------------------------------------------
    % ------- start of reference architecture --------------
    %{
	net.normalization.imageSize = [227, 227, 3] ;
    net.layers = {} ;

    net = iLab_simplenn_addblock(net, opts, '1', 11, 11, 3, 96, 4, 0) ;
    net = iLab_simplenn_addnorm(net, opts, '1') ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', 0) ;


    net = iLab_simplenn_addblock(net, opts, '2', 5, 5, 48, 256, 1, 2) ;
    net = iLab_simplenn_addnorm(net, opts, '2') ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', 0) ;


    net = iLab_simplenn_addblock(net, opts, '3', 3, 3, 256, 384, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '4', 3, 3, 192, 384, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '5', 3, 3, 192, 256, 1, 1) ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = iLab_simplenn_addblock(net, opts, '6', 6, 6, 256, 1024, 1, 0) ;
    net = iLab_simplenn_addDropout(net, opts, '6') ;

    net = iLab_simplenn_addblock(net, opts, '7', 1, 1, 1024, 1024, 1, 0) ;
    net = iLab_simplenn_addDropout(net, opts, '7') ;

    net = iLab_simplenn_addblock(net, opts, '8_1', 1, 1, 1024,  nclasses, 1, 0) ;
    net.layers(end) = [] ;
    if opts.batchNormalization, net.layers(end) = [] ; end
    %}
    
    % ----------end of reference architecture ----------------------
    % -------------------------------------------------------------------
    
    net = dagnn.DagNN();

    normalization.imageSize     = [227, 227, 3] ;
	normalization.border        = 256 - normalization.imageSize(1:2) ;
    normalization.interpolation = 'bicubic' ;
    normalization.averageImage  = [] ;
    normalization.keepAspect    = true ;      
    net.meta.normalization      = normalization;   


    nLayers = 7;
    conv_fnames  = cell(nLayers,1);
    conv_bnames  = cell(nLayers,1);
    bnorm_wnames = cell(nLayers,1);
    bnorm_bnames = cell(nLayers,1);
    
    synLayers = cell(nLayers*10,1); % make sure each layer has less than 7
                                    % dropout layers
    cntDropout = 0;
% >>>>>>>>>>>>>>>>>> start of the for loop to build the parallel streams >>>
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    
    for s=1:nstreams
        
        cntDropout = 0;
        namesStandard = iLab_dagnn_getStandardLayerNames;
        %% layer 1: convolution, activation, normalization and pooling 
    %     nameLayer       = 'conv1';
    %     nameBnorm       = 'bnorm1';
    %     nameNorm        = 'norm1';
    %     namePool        = 'pool1';
    %     nameActivation  = 'relu1';    
        nthLayer = 1;
        nameLayer       = sprintf('%s%d', namesStandard.('conv'),  nthLayer);
        nameBnorm       = sprintf('%s%d', namesStandard.('bnorm'), nthLayer);
        nameNorm        = sprintf('%s%d', namesStandard.('norm'),  nthLayer);
        nameActivation  = sprintf('%s%d', namesStandard.('relu'),  nthLayer);
        namePool        = sprintf('%s%d', namesStandard.('pool'),  nthLayer);
        nameInput       = sprintf('%s%d', namesStandard.('input'), nthLayer);    

        inputs          = iLab_dagnn_getNewLayerName(net, nameInput);    
        nameLayer       = iLab_dagnn_getNewLayerName(net, nameLayer);
        outputs         = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));     

        LayerNames{s}  = cat(2,LayerNames{s}, {nameLayer}); 
        LayerInputs{s} = cat(2,LayerInputs{s}, {inputs});
        
        fname = iLab_dagnn_setParamNameFilter(nameLayer);
        bname = iLab_dagnn_setParamNameBias(nameLayer); 
        if s==1
            conv_fnames{nthLayer} = fname; conv_bnames{nthLayer} = bname;
        end
        if bshare
            fname = conv_fnames{nthLayer};
            bname = conv_bnames{nthLayer};       
        end
        filter = []; bias = [];

        net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                           {'size',            [11 11 3 96], ...
                            'stride',           4, ...
                            'pad',              0, ...
                            'filter',           filter, ...
                            'bias',             bias, ...
                            'fname',            fname, ...
                            'bname',            bname, ...
                            'learningRate',     opts.conv.learningRate, ...
                            'weightDecay',      opts.conv.weightDecay, ...
                            'scale',            opts.conv.scale, ...
                            'weightInitMethod', opts.conv.weightInitMethod});
        
        if opts.batchNormalization      
            inputs     =  outputs;
            nameBnorm  =  iLab_dagnn_getNewLayerName(net, nameBnorm);
            outputs    =  iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));
                
            wname = iLab_dagnn_setParamNameBNmean(nameBnorm);
            bname = iLab_dagnn_setParamNameBias(nameBnorm);
            if s==1
                bnorm_wnames{nthLayer} = wname; bnorm_bnames{nthLayer} = bname;
            end            
            if bshare
                wname = bnorm_wnames{nthLayer};
                bname = bnorm_bnames{nthLayer};
            end
            weight = []; bias = [];

            net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels',    96, ...
                                    'weight',       weight, ...
                                    'bias',         bias, ...
                                    'wname',        wname, ...
                                    'bname',        bname, ...                                    
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay',  opts.bnorm.weightDecay});
        end    

        inputs = outputs;
        nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
        outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
        net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});

        inputs   = outputs;
        nameNorm = iLab_dagnn_getNewLayerName(net, nameNorm);
        outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameNorm));
        net = iLab_dagnn_addlayer_norm(net, nameNorm, inputs, outputs, {'param', opts.norm.param});

        inputs   = outputs;
        namePool = iLab_dagnn_getNewLayerName(net, namePool);
        outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', namePool));
        net = iLab_dagnn_addlayer_pooling(net, namePool, inputs, outputs, ...
                                    {'method', opts.pooling.method, ...
                                    'pool', [3 3], ...
                                    'stride', 2, ...
                                    'pad', 0});
        LayerOutputs{s} = cat(2, LayerOutputs{s}, {outputs});    

        %% layer 2: convolution, activation, normalization and pooling 
    % 	nameLayer       = 'conv2';
    %     nameBnorm       = 'bnorm2';
    %     nameNorm        = 'norm2';
    %     namePool        = 'pool2';
    %     nameActivation  = 'relu2';    
        nthLayer = 2;
        nameLayer       = sprintf('%s%d', namesStandard.('conv'),  nthLayer);
        nameBnorm       = sprintf('%s%d', namesStandard.('bnorm'), nthLayer);
        nameNorm        = sprintf('%s%d', namesStandard.('norm'),  nthLayer);
        nameActivation  = sprintf('%s%d', namesStandard.('relu'),  nthLayer);
        namePool        = sprintf('%s%d', namesStandard.('pool'),  nthLayer);    

        inputs = outputs;
        nameLayer   = iLab_dagnn_getNewLayerName(net, nameLayer);
        outputs     = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));

        LayerNames{s}  = cat(2,LayerNames{s}, {nameLayer}); 
        LayerInputs{s} = cat(2,LayerInputs{s}, {inputs});
        
        fname = iLab_dagnn_setParamNameFilter(nameLayer);
        bname = iLab_dagnn_setParamNameBias(nameLayer); 
        if s==1
            conv_fnames{nthLayer} = fname; conv_bnames{nthLayer} = bname;
        end
        if bshare
            fname = conv_fnames{nthLayer};
            bname = conv_bnames{nthLayer};       
        end
        filter = []; bias = [];
        
        net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                    {'size', [5 5 48 256], 'stride', 1, 'pad', 2, ...
                    'filter',           filter, ...
                    'bias',             bias, ...
                    'fname',            fname, ...
                    'bname',            bname, ...
                    'learningRate',     opts.conv.learningRate, ...
                    'weightDecay',      opts.conv.weightDecay, ...
                    'weightInitMethod', opts.conv.weightInitMethod, ...
                    'scale',            opts.conv.scale});
 
        if opts.batchNormalization        
            inputs = outputs;
            nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
            outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));
      
            wname = iLab_dagnn_setParamNameBNmean(nameBnorm);
            bname = iLab_dagnn_setParamNameBias(nameBnorm);
            if s==1
                bnorm_wnames{nthLayer} = wname; bnorm_bnames{nthLayer} = bname;
            end            
            if bshare
                wname = bnorm_wnames{nthLayer};
                bname = bnorm_bnames{nthLayer};
            end
            weight = []; bias = [];

            net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels',    256, ...
                                    'weight',       weight, ...
                                    'bias',         bias, ...
                                    'wname',        wname, ...
                                    'bname',        bname, ...                                    
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay',  opts.bnorm.weightDecay});  
        end    

        inputs = outputs;
        nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
        outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation)) ;    
        net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});

        inputs = outputs;
        nameNorm = iLab_dagnn_getNewLayerName(net, nameNorm);
        outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameNorm));
        net = iLab_dagnn_addlayer_norm(net, nameNorm, inputs, outputs, {'param', opts.norm.param});

        inputs = outputs;
        namePool = iLab_dagnn_getNewLayerName(net, namePool);
        outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', namePool));
        net = iLab_dagnn_addlayer_pooling(net, namePool,  inputs, outputs, ...
                                {'method', opts.pooling.method, ... 
                                'pool', [3 3], ...
                                'stride', 2,...
                                'pad', 0});

        LayerOutputs{s} = cat(2, LayerOutputs{s}, {outputs});    
        %% layer 3: convolution, activation
    %     nameLayer        = 'conv3';
    %     nameBnorm        = 'bnorm3';
    %     nameActivation   = 'relu3';    
        nthLayer = 3;
        nameLayer       = sprintf('%s%d', namesStandard.('conv'),  nthLayer);
        nameBnorm       = sprintf('%s%d', namesStandard.('bnorm'), nthLayer);    
        nameActivation  = sprintf('%s%d', namesStandard.('relu'),  nthLayer);        

        inputs = outputs;
        nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
        outputs   = iLab_dagnn_getNewVarName(net,  sprintf('%sout', nameLayer));
        LayerNames{s}  = cat(2,LayerNames{s}, {nameLayer}); 
        LayerInputs{s} = cat(2,LayerInputs{s}, {inputs});
                
        fname = iLab_dagnn_setParamNameFilter(nameLayer);
        bname = iLab_dagnn_setParamNameBias(nameLayer); 
        if s==1
            conv_fnames{nthLayer} = fname; conv_bnames{nthLayer} = bname;
        end
        if bshare
            fname = conv_fnames{nthLayer};
            bname = conv_bnames{nthLayer};       
        end
        filter = []; bias = [];
        
        net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                    {'size', [3 3 256 384], 'stride', 1, 'pad', 1, ...
                    'filter',           filter, ...
                    'bias',             bias, ...
                    'fname',            fname, ...
                    'bname',            bname, ...
                    'learningRate',     opts.conv.learningRate, ...
                    'weightDecay',      opts.conv.weightDecay, ...
                    'weightInitMethod', opts.conv.weightInitMethod, ...
                    'scale',            opts.conv.scale});                

        if opts.batchNormalization
            inputs = outputs;
            nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
            outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));

            wname = iLab_dagnn_setParamNameBNmean(nameBnorm);
            bname = iLab_dagnn_setParamNameBias(nameBnorm);
            if s==1
                bnorm_wnames{nthLayer} = wname; bnorm_bnames{nthLayer} = bname;
            end            
            if bshare
                wname = bnorm_wnames{nthLayer};
                bname = bnorm_bnames{nthLayer};
            end
            weight = []; bias = [];

            net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels',    384, ...
                                    'weight',       weight, ...
                                    'bias',         bias, ...
                                    'wname',        wname, ...
                                    'bname',        bname, ...                                    
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay',  opts.bnorm.weightDecay});                                                 
                                                 
        end    

        inputs = outputs;
        nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
        outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
        net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});
        LayerOutputs{s} = cat(2, LayerOutputs{s}, {outputs});    

        %% layer 4: convolution, activation
    % 	nameLayer        = 'conv4';
    %     nameBnorm       = 'bnorm4';
    %     nameActivation  = 'relu4';
        nthLayer = 4;
        nameLayer       = sprintf('%s%d', namesStandard.('conv'),  nthLayer);
        nameBnorm       = sprintf('%s%d', namesStandard.('bnorm'), nthLayer);
        nameActivation  = sprintf('%s%d', namesStandard.('relu'),  nthLayer);

        inputs = outputs;
        nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
        outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));
        LayerNames{s}  = cat(2,LayerNames{s}, {nameLayer}); 
        LayerInputs{s} = cat(2,LayerInputs{s}, {inputs});
                
        fname = iLab_dagnn_setParamNameFilter(nameLayer);
        bname = iLab_dagnn_setParamNameBias(nameLayer); 
        if s==1
            conv_fnames{nthLayer} = fname; conv_bnames{nthLayer} = bname;
        end
        if bshare
            fname = conv_fnames{nthLayer};
            bname = conv_bnames{nthLayer};       
        end
        filter = []; bias = [];
        
        net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                    {'size', [3 3 192 384], 'stride', 1, 'pad', 1, ...
                    'filter',           filter, ...
                    'bias',             bias, ...
                    'fname',            fname, ...
                    'bname',            bname, ...
                    'learningRate',     opts.conv.learningRate, ...
                    'weightDecay',      opts.conv.weightDecay, ...
                    'weightInitMethod', opts.conv.weightInitMethod, ...
                    'scale',            opts.conv.scale});                

        if opts.batchNormalization
            inputs = outputs;
            nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
            outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));

            wname = iLab_dagnn_setParamNameBNmean(nameBnorm);
            bname = iLab_dagnn_setParamNameBias(nameBnorm);
            if s==1
                bnorm_wnames{nthLayer} = wname; bnorm_bnames{nthLayer} = bname;
            end            
            if bshare
                wname = bnorm_wnames{nthLayer};
                bname = bnorm_bnames{nthLayer};
            end
            weight = []; bias = [];

            net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels',    384, ...
                                    'weight',       weight, ...
                                    'bias',         bias, ...
                                    'wname',        wname, ...
                                    'bname',        bname, ...                                    
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay',  opts.bnorm.weightDecay});                                                  
        end    

        inputs = outputs;
        nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
        outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
        net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});    
        LayerOutputs{s} = cat(2, LayerOutputs{s}, {outputs});    

        %% layer 5: convolution, activation, pooling
    % 	nameLayer        = 'conv5';
    %     nameBnorm       = 'bnorm5';
    %     namePool        = 'pool5';
    %     nameActivation  = 'relu5';    
        nthLayer = 5;
        nameLayer       = sprintf('%s%d', namesStandard.('conv'),  nthLayer);
        nameBnorm       = sprintf('%s%d', namesStandard.('bnorm'), nthLayer);
        nameActivation  = sprintf('%s%d', namesStandard.('relu'),  nthLayer);
        namePool        = sprintf('%s%d', namesStandard.('pool'),  nthLayer);    

        inputs = outputs;
        nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
        outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));
        LayerNames{s}  = cat(2,LayerNames{s}, {nameLayer}); 
        LayerInputs{s} = cat(2,LayerInputs{s}, {inputs});
                
        fname = iLab_dagnn_setParamNameFilter(nameLayer);
        bname = iLab_dagnn_setParamNameBias(nameLayer); 
        if s==1
            conv_fnames{nthLayer} = fname; conv_bnames{nthLayer} = bname;
        end
        if bshare
            fname = conv_fnames{nthLayer};
            bname = conv_bnames{nthLayer};       
        end
        filter = []; bias = [];
        
        net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                    {'size', [3 3 192 256], 'stride', 1, 'pad', 1, ...
                    'filter',           filter, ...
                    'bias',             bias, ...
                    'fname',            fname, ...
                    'bname',            bname, ...
                    'learningRate',     opts.conv.learningRate, ...
                    'weightDecay',      opts.conv.weightDecay, ...
                    'weightInitMethod', opts.conv.weightInitMethod, ...
                    'scale',            opts.conv.scale});                

        if opts.batchNormalization
            inputs = outputs;
            nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
            outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));

            wname = iLab_dagnn_setParamNameBNmean(nameBnorm);
            bname = iLab_dagnn_setParamNameBias(nameBnorm);
            if s==1
                bnorm_wnames{nthLayer} = wname; bnorm_bnames{nthLayer} = bname;
            end            
            if bshare
                wname = bnorm_wnames{nthLayer};
                bname = bnorm_bnames{nthLayer};
            end
            weight = []; bias = [];

            net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels',    256, ...
                                    'weight',       weight, ...
                                    'bias',         bias, ...
                                    'wname',        wname, ...
                                    'bname',        bname, ...                                    
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay',  opts.bnorm.weightDecay});                                                  
        end    
        inputs = outputs;
        nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
        outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
        net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type',  'relu'});

        inputs = outputs;
        namePool = iLab_dagnn_getNewLayerName(net, namePool);
        outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', namePool));
        net = iLab_dagnn_addlayer_pooling(net, namePool, inputs, outputs,  ...
                                            {'method', opts.pooling.method,...
                                            'pool', [3 3], ...
                                            'stride', 2, ...
                                            'pad',0});
        LayerOutputs{s} = cat(2, LayerOutputs{s}, {outputs});    

        %% layer 6: fully connected layer: convolution, activation, dropout
    % 	nameLayer       = 'fc1';
    %     nameBnorm       = 'bnorm6';
    %     nameActivation  = 'relu6';
    %     nameDropout     = 'dropout6';   
        nthLayer = 6;
        nameLayer       = sprintf('%s%d', namesStandard.('fc'),     nthLayer);
        nameBnorm       = sprintf('%s%d', namesStandard.('bnorm'),  nthLayer);
        nameActivation  = sprintf('%s%d', namesStandard.('relu'),   nthLayer);
        nameDropout     = sprintf('%s%d', namesStandard.('dropout'),nthLayer);

        inputs = outputs;
        nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
        outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));

        LayerNames{s}  = cat(2,LayerNames{s}, {nameLayer}); 
        LayerInputs{s} = cat(2,LayerInputs{s}, {inputs});
                
        fname = iLab_dagnn_setParamNameFilter(nameLayer);
        bname = iLab_dagnn_setParamNameBias(nameLayer); 
        if s==1
            conv_fnames{nthLayer} = fname; conv_bnames{nthLayer} = bname;
        end
        if bshare
            fname = conv_fnames{nthLayer};
            bname = conv_bnames{nthLayer};       
        end
        filter = []; bias = [];
        
        net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                    {'size', [6 6 256 fcsize], 'stride', 1, 'pad', 0, ...
                    'filter',           filter, ...
                    'bias',             bias, ...
                    'fname',            fname, ...
                    'bname',            bname, ...
                    'learningRate',     opts.conv.learningRate, ...
                    'weightDecay',      opts.conv.weightDecay, ...
                    'weightInitMethod', opts.conv.weightInitMethod, ...
                    'scale',            opts.conv.scale});                 

        if opts.batchNormalization
            inputs = outputs;
            nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
            outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));
            
            wname = iLab_dagnn_setParamNameBNmean(nameBnorm);
            bname = iLab_dagnn_setParamNameBias(nameBnorm);
            if s==1
                bnorm_wnames{nthLayer} = wname; bnorm_bnames{nthLayer} = bname;
            end            
            if bshare
                wname = bnorm_wnames{nthLayer};
                bname = bnorm_bnames{nthLayer};
            end
            weight = []; bias = [];

            net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels',    fcsize, ...
                                    'weight',       weight, ...
                                    'bias',         bias, ...
                                    'wname',        wname, ...
                                    'bname',        bname, ...                                    
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay',  opts.bnorm.weightDecay});                                                 
        end    

        inputs = outputs;
        nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
        outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
        net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});

        inputs = outputs;
        nameDropout = iLab_dagnn_getNewLayerName(net, nameDropout);
        outputs     = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameDropout));
        net = iLab_dagnn_addlayer_dropout(net, nameDropout, inputs, outputs, {'rate', opts.dropout.rate});

        cntDropout = cntDropout + 1;
        l = net.getLayerIndex(nameDropout);
        synLayers{cntDropout} = cat(2, synLayers{cntDropout}, l);
        
        LayerOutputs{s} = cat(2, LayerOutputs{s}, {outputs});    
        %% layer 7: fully connected layer: convolution, activation, dropout
    % 	nameLayer        = 'fc2';
    %     nameBnorm       = 'bnorm7';
    %     nameActivation  = 'relu7';
    %     nameDropout     = 'dropout7';   
        nthLayer = 7;
        nameLayer       = sprintf('%s%d', namesStandard.('fc'),     nthLayer);
        nameBnorm       = sprintf('%s%d', namesStandard.('bnorm'),  nthLayer);
        nameActivation  = sprintf('%s%d', namesStandard.('relu'),   nthLayer);
        nameDropout     = sprintf('%s%d', namesStandard.('dropout'),nthLayer);    

        inputs = outputs;
        nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
        outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));

        LayerNames{s}  = cat(2,LayerNames{s}, {nameLayer}); 
        LayerInputs{s} = cat(2,LayerInputs{s}, {inputs});
                
        fname = iLab_dagnn_setParamNameFilter(nameLayer);
        bname = iLab_dagnn_setParamNameBias(nameLayer); 
        if s==1
            conv_fnames{nthLayer} = fname; conv_bnames{nthLayer} = bname;
        end
        if bshare
            fname = conv_fnames{nthLayer};
            bname = conv_bnames{nthLayer};       
        end
        filter = []; bias = [];
        
        net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                    {'size', [1 1 fcsize fcsize], 'stride', 1, 'pad', 0, ...
                    'filter',           filter, ...
                    'bias',             bias, ...
                    'fname',            fname, ...
                    'bname',            bname, ...
                    'learningRate',     opts.conv.learningRate, ...
                    'weightDecay',      opts.conv.weightDecay, ...
                    'weightInitMethod', opts.conv.weightInitMethod, ...
                    'scale',            opts.conv.scale});                 

        if opts.batchNormalization
            inputs = outputs;
            nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
            outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));

            wname = iLab_dagnn_setParamNameBNmean(nameBnorm);
            bname = iLab_dagnn_setParamNameBias(nameBnorm);
            if s==1
                bnorm_wnames{nthLayer} = wname; bnorm_bnames{nthLayer} = bname;
            end            
            if bshare
                wname = bnorm_wnames{nthLayer};
                bname = bnorm_bnames{nthLayer};
            end
            weight = []; bias = [];

            net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels',    fcsize, ...
                                    'weight',       weight, ...
                                    'bias',         bias, ...
                                    'wname',        wname, ...
                                    'bname',        bname, ...                                    
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay',  opts.bnorm.weightDecay});                                                  
        end  

        inputs = outputs;
        nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
        outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
        net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});

        inputs = outputs;
        nameDropout = iLab_dagnn_getNewLayerName(net, nameDropout);
        outputs     = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameDropout));
        net = iLab_dagnn_addlayer_dropout(net, nameDropout, inputs, outputs, {'rate', opts.dropout.rate});

        cntDropout = cntDropout + 1;
        l = net.getLayerIndex(nameDropout);
        synLayers{cntDropout} = cat(2, synLayers{cntDropout}, l);
        
        LayerOutputs{s} = cat(2, LayerOutputs{s}, {outputs});    
   
    
    end 
% >>>>>>>>>>>>>>>>>> end of the for loop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    mapsInputs = cell(nstreams,1);
    mapsOutputs = cell(nstreams,1);
    
    for s=1:nstreams
        mapsInputs{s}   =   containers.Map(LayerNames{s}, LayerInputs{s});
        mapsOutputs{s}  =   containers.Map(LayerNames{s}, LayerOutputs{s});
    end
    
    net.synLayers = cat(2, net.synLayers, synLayers(1:cntDropout));
    
    varargout = {net, LayerNames, mapsInputs, mapsOutputs};
    
    
end