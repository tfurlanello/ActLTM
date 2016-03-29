function varargout = iLab_dagnn_vgg_m(nclasses, args)
%% note:
% the output of the architecture is a prediction vector
% don't add the loss layer as the last layer; we will have much more
% flexibilities, since we could add manually-crafted loss functions later
% on
    opts.batchNormalization = false ;	
    
    opts.conv.weightInitMethod = 'gaussian';
    opts.conv.scale  = 1.0;
    opts.conv.learningRate  = [1 2];
    opts.conv.weightDecay   = [1 0];
    opts.fc.size            = 1024;
    
    opts.bnorm.learningRate = [2 1];
    opts.bnorm.weightDecay  = [0 0];
    
    opts.norm.param     = [5 1 0.0001/5 0.75];
    opts.pooling.method = 'max';
    opts.dropout.rate   = 0.5;
    
    opts = vl_argparse(opts, args) ;
    
    LayerNames      = {};
    LayerInputs     = {};
    LayerOutputs    = {};
    
    fcsize = opts.fc.size;
%---------------------------------------------------------------------
% ------------------ start of the reference architecture--------------
    %{
    net.layers = {} ;
    net = add_block(net, opts, '1', 7, 7, 3, 96, 2, 0) ;
    net = add_norm(net, opts, '1') ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = add_block(net, opts, '2', 5, 5, 96, 256, 2, 1) ;
    net = add_norm(net, opts, '2') ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', [0 1 0 1]) ;

    net = add_block(net, opts, '3', 3, 3, 256, 512, 1, 1) ;
    net = add_block(net, opts, '4', 3, 3, 512, 512, 1, 1) ;
    net = add_block(net, opts, '5', 3, 3, 512, 512, 1, 1) ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = add_block(net, opts, '6', 6, 6, 512, 1024, 1, 0) ;
    net = add_dropout(net, opts, '6') ;

    net = add_block(net, opts, '7', 1, 1, 1024, 1024, 1, 0) ;
    net = add_dropout(net, opts, '7') ;

    net = add_block(net, opts, '8', 1, 1, 1024,   nclasses, 1, 0) ;
    net.layers(end) = [] ;
    if opts.batchNormalization, net.layers(end) = [] ; end
    %}
%------------------- end of the reference architecture----------------
%---------------------------------------------------------------------           
    net = dagnn.DagNN();
    
    normalization.imageSize     = [224, 224, 3] ;
    normalization.border        = 256 - normalization.imageSize(1:2) ;
    normalization.interpolation = 'bicubic' ;
    normalization.averageImage  = [] ;
    normalization.keepAspect    = true ;      
    
	net.meta.normalization  = normalization;    

    %% layer 1: convolution, activation, normalization and pooling
	nameConv        = 'conv1';
    nameBnorm       = 'bnorm1';
    nameNorm        = 'norm1';
    namePool        = 'pool1';
    nameActivation  = 'relu1';
    
    inputs = 'input';
    outputs = sprintf('%sout', nameConv);
    LayerNames = cat(2, LayerNames, nameConv);
    LayerInputs = cat(2, LayerInputs, inputs);
    net = iLab_dagnn_addlayer_conv(net, nameConv, inputs, outputs, ...
                                   {'size', [7  7 3 96], ...
                                   'stride', 2, 'pad', 0, ...
                                   'learningRate', opts.conv.learningRate, ...
                                   'weightDecay', opts.conv.weightDecay,...
                                   'weightInitMethod', opts.conv.weightInitMethod, ...
                                   'scale', opts.conv.scale});
    if opts.batchNormalization
        inputs = outputs;
        nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
        outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));
        net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels', 96, ...
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay', opts.bnorm.weightDecay});
    end
    
    inputs = outputs;
    nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
    outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
    net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});
    
    inputs = outputs;
    nameNorm = iLab_dagnn_getNewLayerName(net, nameNorm);
    outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameNorm));
    net = iLab_dagnn_addlayer_norm(net, nameNorm, inputs, outputs, {'param', opts.norm.param});
    
    inputs = outputs;
    namePool = iLab_dagnn_getNewLayerName(net, namePool);
    outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', namePool));
    net = iLab_dagnn_addlayer_pooling(net, namePool, inputs, outputs, ...
                                     {'method', opts.pooling.method, ...
                                     'pool', [3 3], ...
                                     'stride', 2, ...
                                     'pad', 0});
                                 
    LayerOutputs = cat(2, LayerOutputs, outputs);                                 
    %% layer 2: convolution, activation, normalization and pooling
    nameConv        = 'conv2';
    nameBnorm       = 'bnorm2';
    nameNorm        = 'norm2';
    namePool        = 'pool2';
    nameActivation  = 'relu2';    
    
    inputs = outputs;
    nameConv = iLab_dagnn_getNewLayerName(net, nameConv);
    outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameConv));
    
    LayerNames = cat(2, LayerNames, nameConv);
    LayerInputs = cat(2, LayerInputs, inputs);
    net = iLab_dagnn_addlayer_conv(net, nameConv, inputs, outputs, ...
                                   {'size', [5 5 96 256], ...
                                   'stride', 2, 'pad', 1, ...
                                   'learningRate', opts.conv.learningRate, ...
                                   'weightDecay', opts.conv.weightDecay,...
                                   'weightInitMethod', opts.conv.weightInitMethod, ...
                                   'scale', opts.conv.scale});
    if opts.batchNormalization
        inputs = outputs;
        nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
        outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));
        net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels', 256, ...
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay', opts.bnorm.weightDecay});
    end
    
    inputs = outputs;
    nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
    outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
    net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});
    
    inputs = outputs;
    nameNorm = iLab_dagnn_getNewLayerName(net, nameNorm);
    outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameNorm));
    net = iLab_dagnn_addlayer_norm(net, nameNorm, inputs, outputs, {'param', opts.norm.param});
    
    inputs = outputs;
    namePool = iLab_dagnn_getNewLayerName(net, namePool);
    outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', namePool));
    net = iLab_dagnn_addlayer_pooling(net, namePool, inputs, outputs, ...
                                     {'method', opts.pooling.method, ...
                                     'pool', [3 3], ...
                                     'stride', 2, ...
                                     'pad', [0 1 0 1]});  
                                 
    LayerOutputs = cat(2, LayerOutputs, outputs);                                 
    
    %% layer 3: convolution, activation
	nameConv        = 'conv3';
    nameBnorm       = 'bnorm3';
    nameActivation  = 'relu3';
    
    inputs = outputs;
    nameConv  = iLab_dagnn_getNewLayerName(net, nameConv);    
    outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameConv));
    
    LayerNames = cat(2, LayerNames, nameConv);
    LayerInputs = cat(2, LayerInputs, inputs);
    net = iLab_dagnn_addlayer_conv(net, nameConv, inputs, outputs, ...
                                   {'size', [3 3 256 512], ...
                                   'stride', 1, 'pad', 1, ...
                                   'learningRate', opts.conv.learningRate, ...
                                   'weightDecay', opts.conv.weightDecay,...
                                   'weightInitMethod', opts.conv.weightInitMethod, ...
                                   'scale', opts.conv.scale});
    if opts.batchNormalization
        inputs = outputs;
        nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
        outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));
        net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels', 512, ...
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay', opts.bnorm.weightDecay});
    end
    
    inputs = outputs;
    nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
    outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
    net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});    
    
    LayerOutputs = cat(2, LayerOutputs, outputs);
    %% layer 4: convolution, activation
    nameConv        = 'conv4';
    nameBnorm       = 'bnorm4';
    nameActivation  = 'relu4';    
    
    inputs = outputs;
    nameConv  = iLab_dagnn_getNewLayerName(net, nameConv);
    outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameConv));
    
    LayerNames = cat(2, LayerNames, nameConv);
    LayerInputs = cat(2, LayerInputs, inputs);
    net = iLab_dagnn_addlayer_conv(net, nameConv, inputs, outputs, ...
                                   {'size', [3 3 512 512], ...
                                   'stride', 1, 'pad', 1, ...
                                   'learningRate', opts.conv.learningRate, ...
                                   'weightDecay', opts.conv.weightDecay,...
                                   'weightInitMethod', opts.conv.weightInitMethod, ...
                                   'scale', opts.conv.scale});
    if opts.batchNormalization
        inputs = outputs;
        nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
        outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));
        net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels', 512, ...
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay', opts.bnorm.weightDecay});
    end
    
    inputs = outputs;
    nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
    outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
    net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type',  'relu'});   
    
    LayerOutputs = cat(2, LayerOutputs, outputs);
    %% layer 5: convolution, activation, pooling
    nameConv        = 'conv5';
    nameBnorm       = 'bnorm5';
    namePool        = 'pool5';
    nameActivation  = 'relu5';    
    
    inputs = outputs;
    nameConv = iLab_dagnn_getNewLayerName(net, nameConv);
    outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameConv));
    
    LayerNames = cat(2, LayerNames, nameConv);
    LayerInputs = cat(2, LayerInputs, inputs);
    net = iLab_dagnn_addlayer_conv(net, nameConv, inputs, outputs, ...
                                   {'size', [3 3 512 512], ...
                                   'stride', 1, 'pad', 1, ...
                                   'learningRate', opts.conv.learningRate, ...
                                   'weightDecay', opts.conv.weightDecay,...
                                   'weightInitMethod', opts.conv.weightInitMethod, ...
                                   'scale', opts.conv.scale});
    if opts.batchNormalization
        inputs = outputs;
        nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
        outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));
        net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels', 512, ...
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay', opts.bnorm.weightDecay});
    end
    inputs = outputs;
    nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
    outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
    net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});
    
    inputs = outputs;
    namePool = iLab_dagnn_getNewLayerName(net, namePool);
    outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', namePool));
    net = iLab_dagnn_addlayer_pooling(net, namePool, inputs, outputs, ...
                                    {'method', opts.pooling.method, ...
                                    'pool', [3 3], ...
                                    'stride', 2,...
                                    'pad', 0});

    LayerOutputs = cat(2, LayerOutputs, outputs);
    %% layer 6, fully connected layer: convolution, activation, dropout
    nameConv        = 'fc1';
    nameBnorm       = 'bnorm6';
    nameActivation  = 'relu6';    
    nameDropout     = 'dropout6';
    
    inputs = outputs;
    nameConv = iLab_dagnn_getNewLayerName(net, nameConv);
    outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameConv));
    
    LayerNames = cat(2, LayerNames, nameConv);
    LayerInputs = cat(2, LayerInputs, inputs);
    net = iLab_dagnn_addlayer_conv(net, nameConv, inputs, outputs, ...
                                   {'size', [6 6 512 fcsize], ...
                                   'stride', 1, 'pad', 0, ...
                                   'learningRate', opts.conv.learningRate, ...
                                   'weightDecay', opts.conv.weightDecay,...
                                   'weightInitMethod', opts.conv.weightInitMethod, ...
                                   'scale', opts.conv.scale});
    if opts.batchNormalization
        inputs = outputs;
        nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
        outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));
        net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                    {'nchannels', fcsize, ...
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay', opts.bnorm.weightDecay});
    end
    inputs = outputs;
    nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
    outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
    net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});  
    
    inputs = outputs;
    nameDropout  = iLab_dagnn_getNewLayerName(net, nameDropout);
    outputs      = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameDropout));
    net = iLab_dagnn_addlayer_dropout(net, nameDropout, inputs, outputs, {'rate', opts.dropout.rate});
    
    LayerOutputs = cat(2, LayerOutputs, outputs);
    %% layer 7, fully connected layer: convolution, activation, dropout
    nameConv        = 'fc2';
    nameBnorm       = 'bnorm7';
    nameActivation  = 'relu7';    
    nameDropout     = 'dropout7';
    
    inputs = outputs;
    nameConv = iLab_dagnn_getNewLayerName(net, nameConv);
    outputs  = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameConv));
    
    LayerNames = cat(2, LayerNames, nameConv);
    LayerInputs = cat(2, LayerInputs, inputs);
    net = iLab_dagnn_addlayer_conv(net, nameConv, inputs, outputs, ...
                                   {'size', [1 1 fcsize fcsize], ...
                                   'stride', 1, 'pad', 0, ...
                                   'learningRate', opts.conv.learningRate, ...
                                   'weightDecay', opts.conv.weightDecay,...
                                   'weightInitMethod', opts.conv.weightInitMethod, ...
                                   'scale', opts.conv.scale});
    if opts.batchNormalization
        inputs = outputs;
        nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);        
        outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));
        net = iLab_dagnn_addlayer_bnorm(net, nameBnorm,  inputs, outputs, ...
                                    {'nchannels', fcsize, ...
                                    'learningRate', opts.bnorm.learningRate, ...
                                    'weightDecay', opts.bnorm.weightDecay});
    end
    
    inputs = outputs;
    nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
    outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
    net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});  
    
    inputs = outputs;
    nameDropout  = iLab_dagnn_getNewLayerName(net, nameDropout);
    outputs      = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameDropout));
    net = iLab_dagnn_addlayer_dropout(net, nameDropout, inputs, outputs, {'rate', opts.dropout.rate});    
    
    LayerOutputs = cat(2, LayerOutputs, outputs);
    %% layer 8, prediction layer: convolution
    nameLayer = 'prediction';
    
    inputs = outputs;
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
    outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));
    
    LayerNames = cat(2, LayerNames, nameLayer);
    LayerInputs = cat(2, LayerInputs, inputs);
    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                {'size', [1 1 fcsize  nclasses], 'stride', 1, 'pad', 0,...
                'learningRate', opts.conv.learningRate, ...
                'weightDecay',  opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale', opts.conv.scale});   
            
    LayerOutputs = cat(2, LayerOutputs, outputs);
    % final touches
    switch lower(opts.conv.weightInitMethod)
      case {'xavier', 'xavierimproved'}
        lindex = net.getLayerIndex(nameLayer);
        params = net.layers(lindex).params;
        pindex = net.getParamIndex(params{1});
        net.params(pindex).value = net.params(pindex).value / 10;    
    end
 
    %% objective layer
%     nameLayer = 'loss';
%     inputs = {outputs, 'label'};
%     outputs = 'objective';
%     net = iLab_dagnn_addlayer_loss(net, nameLayer, inputs, outputs, ...
%                         {'type', 'softmaxlog', ...
%                         'isstructured', false, ...
%                         'labelgraph', []});
    
    mapInputs = containers.Map(LayerNames, LayerInputs);
    mapOutputs = containers.Map(LayerNames, LayerOutputs);
    
    varargout = {net, LayerNames, mapInputs, mapOutputs};
          
end