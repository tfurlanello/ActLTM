function net = iLab_simplenn_alexnet2(nclasses, args)  

    opts.batchNormalization = false ;
     
    opts.conv.weightInitMethod = 'gaussian';
    opts.conv.scale  = 1.0;
    opts.conv.learningRate  = [1 2];
    opts.conv.weightDecay   = [1 0];
    
    opts.bnorm.learningRate = [2 1];
    opts.bnorm.weightDecay  = [0 0];
    
    opts.norm.param     = [5 1 0.0001/5 0.75];
    opts.pooling.method = 'max';
    opts.dropout.rate   = 0.5;
    
    opts = vl_argparse(opts, args) ;
     
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
    

    net.normalization.imageSize     = [227, 227, 3] ;
 	net.normalization.border        = 256 - net.normalization.imageSize(1:2) ;
    net.normalization.interpolation = 'bicubic' ;
    net.normalization.averageImage  = [] ;
    net.normalization.keepAspect    = true ;   
    net.layers = {} ;

    %% layer 1: convolution, activation, normalization and pooling  
    nameConv        = 'conv1';
    nameBnorm       = 'bnorm1';
    nameNorm        = 'norm1';
    namePool        = 'pool1';
    nameActivation  = 'relu1';
    net = iLab_simplenn_addlayer_block(net, nameConv, ...
                {'h', 11, 'w', 11, 'in', 3, 'out', 96, 'stride', 4, 'pad', 0, ...
                'learningRate', opts.conv.learningRate, ...
                'weightDecay',  opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale', opts.conv.scale});
            
    if opts.batchNormalization
      net.layers{end+1} = iLab_simplenn_addlayer_bnorm(net, nameBnorm, ...
                                {'nchannel', 96, 'learningRate', opts.bnorm.learningRate, ...
                                                 'weightDecay',  opts.bnorm.weightDecay});
    end    
    net = iLab_simplenn_addlayer_activation(net, nameActivation, 'relu');
    net = iLab_simplenn_addlayer_norm(net, nameNorm, {'param', opts.norm.param});
    net = iLab_simplenn_addlayer_pooling(net, namePool, ...
                            {'method', opts.pooling.method, ... 
                            'pool', [3 3], ...
                            'stride', 2,...
                            'pad', 0});
                        
    %% layer 2: convolution, activation, normalization and pooling 
	nameConv        = 'conv2';
    nameBnorm       = 'bnorm2';
    nameNorm        = 'norm2';
    namePool        = 'pool2';
    nameActivation  = 'relu2';
    net = iLab_simplenn_addlayer_block(net, nameConv, ...
                {'h', 5, 'w', 5, 'in', 48, 'out', 256, 'stride', 1, 'pad', 2, ...
                'learningRate', opts.conv.learningRate, ...
                'weightDecay',  opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale', opts.conv.scale});
            
    if opts.batchNormalization
      net.layers{end+1} = iLab_simplenn_addlayer_bnorm(net, nameBnorm, ...
                                {'nchannel', 256, 'learningRate', opts.bnorm.learningRate, ...
                                                 'weightDecay',  opts.bnorm.weightDecay});
    end    
    net = iLab_simplenn_addlayer_activation(net, nameActivation, 'relu');
    net = iLab_simplenn_addlayer_norm(net, nameNorm, {'param', opts.norm.param});
    net = iLab_simplenn_addlayer_pooling(net, namePool, ...
                            {'method', opts.pooling.method, ... 
                            'pool', [3 3], ...
                            'stride', 2,...
                            'pad', 0});

    %% layer 3: convolution, activation
	nameConv        = 'conv3';
    nameBnorm       = 'bnorm3';
    nameActivation  = 'relu3';
	net = iLab_simplenn_addlayer_block(net, nameConv, ...
                {'h', 3, 'w', 3, 'in', 256, 'out', 384, 'stride', 1, 'pad', 1, ...
                'learningRate', opts.conv.learningRate, ...
                'weightDecay',  opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale', opts.conv.scale});
            
    if opts.batchNormalization
      net.layers{end+1} = iLab_simplenn_addlayer_bnorm(net, nameBnorm, ...
                                {'nchannel', 384, 'learningRate', opts.bnorm.learningRate, ...
                                                 'weightDecay',  opts.bnorm.weightDecay});
    end    
    net = iLab_simplenn_addlayer_activation(net, nameActivation, 'relu');


    %% layer 4: convolution, activation
	nameConv        = 'conv4';
    nameBnorm       = 'bnorm4';
    nameActivation  = 'relu4';
    net = iLab_simplenn_addlayer_block(net, nameConv, ...
                {'h', 3, 'w', 3, 'in', 192, 'out', 384, 'stride', 1, 'pad', 1, ...
                'learningRate', opts.conv.learningRate, ...
                'weightDecay',  opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale', opts.conv.scale});
            
    if opts.batchNormalization
      net.layers{end+1} = iLab_simplenn_addlayer_bnorm(net, nameBnorm, ...
                                {'nchannel', 384, 'learningRate', opts.bnorm.learningRate, ...
                                                 'weightDecay',  opts.bnorm.weightDecay});
    end    
    net = iLab_simplenn_addlayer_activation(net, nameActivation, 'relu');    
    
    
    %% layer 5: convolution, activation, pooling
	nameConv        = 'conv5';
    nameBnorm       = 'bnorm5';
    namePool        = 'pool5';
    nameActivation  = 'relu5';
	net = iLab_simplenn_addlayer_block(net, nameConv, ...
                {'h', 3, 'w', 3, 'in', 192, 'out', 256, 'stride', 1, 'pad', 1, ...
                'learningRate', opts.conv.learningRate, ...
                'weightDecay',  opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale', opts.conv.scale});
            
    if opts.batchNormalization
      net.layers{end+1} = iLab_simplenn_addlayer_bnorm(net, nameBnorm, ...
                                {'nchannel', 256, 'learningRate', opts.bnorm.learningRate, ...
                                                 'weightDecay',  opts.bnorm.weightDecay});
    end    
    net = iLab_simplenn_addlayer_activation(net, nameActivation, 'relu');
    net = iLab_simplenn_addlayer_pooling(net, namePool, ...
                                        {'method', opts.pooling.method,...
                                        'pool', [3 3], ...
                                        'stride', 2, ...
                                        'pad',0});
    
    %% layer 6: fully connected layer: convolution, activation, dropout
	nameConv        = 'fc1';
    nameBnorm       = 'bnorm6';
    nameActivation  = 'relu6';
    nameDropout     = 'dropout6';
	net = iLab_simplenn_addlayer_block(net, nameConv, ...
                {'h', 6, 'w', 6, 'in', 256, 'out', 1024, 'stride', 1, 'pad', 0, ...
                'learningRate', opts.conv.learningRate, ...
                'weightDecay',  opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale', opts.conv.scale});
            
    if opts.batchNormalization
      net.layers{end+1} = iLab_simplenn_addlayer_bnorm(net, nameBnorm, ...
                                {'nchannel', 1024, 'learningRate', opts.bnorm.learningRate, ...
                                                 'weightDecay',  opts.bnorm.weightDecay});
    end    
    net = iLab_simplenn_addlayer_activation(net, nameActivation, 'relu');
    net = iLab_simplenn_addlayer_dropout(net, nameDropout, opts.dropout.rate);
    
    %% layer 7: fully connected layer: convolution, activation, dropout
	nameConv        = 'fc2';
    nameBnorm       = 'bnorm7';
    nameActivation  = 'relu7';
    nameDropout     = 'dropout7';    
    net = iLab_simplenn_addlayer_block(net, nameConv, ...
                {'h', 1, 'w', 1, 'in', 1024, 'out', 1024, 'stride', 1, 'pad', 0, ...
                'learningRate', opts.conv.learningRate, ...
                'weightDecay',  opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale', opts.conv.scale});
            
    if opts.batchNormalization
      net.layers{end+1} = iLab_simplenn_addlayer_bnorm(net, nameBnorm, ...
                                {'nchannel', 1024, 'learningRate', opts.bnorm.learningRate, ...
                                                 'weightDecay',  opts.bnorm.weightDecay});
    end    
    net = iLab_simplenn_addlayer_activation(net, nameActivation, 'relu');
    net = iLab_simplenn_addlayer_dropout(net, nameDropout, opts.dropout.rate);
  
    %% prediction layer:
    nameConv = 'prediction';
    net = iLab_simplenn_addlayer_block(net, nameConv, ...
                {'h', 1, 'w', 1, 'in', 1024, 'out',  nclasses, 'stride', 1, 'pad', 0, ...
                'learningRate', opts.conv.learningRate, ...
                'weightDecay',  opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale', opts.conv.scale});
    

    % final touches
    switch lower(opts.conv.weightInitMethod)
      case {'xavier', 'xavierimproved'}
        net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
    end
    net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

 
    
end