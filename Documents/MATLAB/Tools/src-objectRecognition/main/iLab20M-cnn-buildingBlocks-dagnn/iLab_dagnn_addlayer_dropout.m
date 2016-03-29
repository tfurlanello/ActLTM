function net = iLab_dagnn_addlayer_dropout(net, nameLayer, inputs, outputs, opts)
    if ~isa(net, 'dagnn.DagNN')
        error('wrong type of network architecture\n');
    end
    
    val_args = struct('rate', 0.5, ...
                  'frozen', false);
              
    opts = iLab_arg2struct(opts);
    opts = iLab_dagnn_validateDropoutParam(opts);
    
    block = dagnn.DropOut() ;
    if isfield(opts,'rate')
        block.rate = opts.rate ;
    end
    if isfield(opts,'frozen')
        block.frozen = opts.frozen ;
    end 
     
    net.addLayer( ...
            nameLayer, ...
            block, ...
            inputs, ...
            outputs);
     
end