function net = iLab_dagnn_addlayer_activation(net, nameLayer, inputs, outputs, opts)
        
    if ~isa(net, 'dagnn.DagNN')
        error('wrong network architectures\n');
    end

    args = struct('type', 'relu');    
    opts = vl_argparse(args, opts);
    
    switch opts.type
        case 'relu'
            block = dagnn.ReLU('opts', {}) ;
        case 'sigmoid'
            block = dagnn.Sigmoid();
        otherwise
            error('unknown activation function\n');
    end

    
    net.addLayer(...
         nameLayer, ...
         block, ...
         inputs, ...
         outputs);
     
end