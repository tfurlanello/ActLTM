function net = iLab_simplenn_addlayer_norm(net, name, opts)
    
    opts = iLab_simplenn_validateNormParam(opts);
    param = opts.param;
    net.layers{end+1} = struct('type', 'normalize', ...
                             'name', name, ...
                             'param', param) ;
end