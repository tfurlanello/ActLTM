function net = iLab_simplenn_addlayer_bnorm(net, name, opts)

	opts      = iLab_simplenn_validateBnormParam(opts);
    nchannels = opts.nchannels;
    weight    = opts.weight;
    bias      = opts.bias;
    learningRate = opts.learningRate;
    weightDecay = opts.weightDecay;
    
    net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('%s',name), ...
                             'weights', {{weight, bias}}, ...
                             'learningRate', learningRate, ...
                             'weightDecay', weightDecay) ;

end