function net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, opts)

    args = struct(...
            'size',         [5 5 48 96],... % h, w, in, out
            'stride',       1,      ...
            'pad',          0,      ...
            'filter',       [],     ...
            'bias',         [],     ...
            'fname',        '',     ...
            'bname',        '',     ...
            'learningRate', [1 2],  ...
            'weightDecay',  [1 0],  ...
            'scale',        1,      ...
            'weightInitMethod', 'gaussian');
     
    if ~isa(net, 'dagnn.DagNN')
        error('wrong type of network architecture\n');
    end
        
    opts = iLab_arg2struct(opts);
%     opts = vl_argparse(args, opts);
    opts = iLab_dagnn_validateConvLayerParam(opts);  
    
    size    = opts.size;
    h = size(1); w = size(2); in = size(3); out = size(4);
    stride  = opts.stride; 
    pad     = opts.pad;
    filter  = opts.filter;
    bias    = opts.bias;
    fname   = opts.fname;
    bname   = opts.bname;
    learningRate = opts.learningRate;
    weightDecay = opts.weightDecay;
    
    if isempty(filter)     
        filter = iLab_nn_initWeight(opts, h, w, in, out, 'single');
    end    
    if isempty(bias)
       bias = zeros(out,1, 'single');
    end   
    
    if isempty(fname)
        fname = iLab_dagnn_setParamNameFilter(nameLayer);
    end    
    if isempty(bname)
        bname = iLab_dagnn_setParamNameBias(nameLayer);
    end
  
    params = struct(...
        'name',         {fname, bname}, ...
        'value',        {filter, bias}, ...
        'learningRate', {learningRate(1), learningRate(2)}, ...
        'weightDecay',  {weightDecay(1), weightDecay(2)}) ;    
    
%     params(1).name = sprintf('%sf', nameLayer);
%     params(2).name = sprintf('%sb', nameLayer);
%     params(1).value = filter;
%     params(2).value = bias;
%     params(1).learningRate = learningRate(1);
%     params(2).learningRate = learningRate(2);
%     params(1).weightDecay  = weightDecay(1);
%     params(2).wieghtDecay   = weightDecay(2);

    block           = dagnn.Conv();
    block.size      = size;
    block.stride    = stride;
    block.pad       = pad;
  
    
	net.addLayer(...
                nameLayer, ...
                block, ...
                inputs, ...
                outputs, ...
                {params.name}) ;
            
    findex = net.getParamIndex(params(1).name);
    bindex = net.getParamIndex(params(2).name);
    
    net.params(findex).value        = filter;
    net.params(bindex).value        = bias;
    net.params(findex).learningRate = learningRate(1);
    net.params(bindex).learningRate = learningRate(2);
    net.params(findex).weightDecay  = weightDecay(1);
    net.params(bindex).weightDecay  = weightDecay(2);
    
    
end