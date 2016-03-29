function [net, new_nameLayer, new_outputs] = ...
            iLab_dagnn_addlayer_norm(net, nameLayer, inputs, outputs, opts)

    if ~isa(net, 'dagnn.DagNN')
        error('wrong type of network architecture\n');
    end
    
    val_opts = struct('param', [5 1 0.0001/5 0.75]);
    opts = iLab_arg2struct(opts);
    opts = iLab_dagnn_validateNormParam(opts);

    block = dagnn.LRN() ;
    if isfield(opts,'param')
        block.param = opts.param ;
    end
    
    net.addLayer(...
            nameLayer, ...
            block, ...
            inputs, ...
            outputs);
end