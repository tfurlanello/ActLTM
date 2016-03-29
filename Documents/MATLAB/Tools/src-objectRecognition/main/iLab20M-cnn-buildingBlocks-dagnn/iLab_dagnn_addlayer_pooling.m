function net = iLab_dagnn_addlayer_pooling(net, nameLayer, inputs, outputs, opts)
    
    if ~isa(net, 'dagnn.DagNN')
        error('wrong type of network architecture\n');
    end
    
    val_opts = struct( ...
                'method', 'max', ...
                'pool', [3 3], ...
                'stride', 2, ...
                'pad', 0);
    
    opts = iLab_arg2struct(opts);
    opts = iLab_dagnn_validatePoolingParam(opts);

    block = dagnn.Pooling() ;
    if isfield(opts,'method')
        block.method = opts.method ;
    end
    if isfield(opts,'pool')
        block.poolSize = opts.pool ;
    end
    if isfield(opts,'pad')
        block.pad = opts.pad ;
    end
    if isfield(opts,'stride')
        block.stride = opts.stride ;
    end

    
    net.addLayer(...
            nameLayer, ...
            block, ...
            inputs, ....
            outputs);
    
end