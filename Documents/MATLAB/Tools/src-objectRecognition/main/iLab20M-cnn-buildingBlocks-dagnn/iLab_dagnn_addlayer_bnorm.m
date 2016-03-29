function net = iLab_dagnn_addlayer_bnorm(net, nameLayer, inputs, outputs, opts)
      
    if ~isa(net, 'dagnn.DagNN')
        error('wrong type of network architectures\n');
    end
    
    opts = iLab_arg2struct(opts);
    opts = iLab_dagnn_validateBnormParam(opts);
    
    learningRate = opts.learningRate;
    weightDecay  = opts.weightDecay;
    nchannels    = opts.nchannels;
    weight = opts.weight;
    bias = opts.bias;

    if isempty(opts.weight)
        weight = ones(nchannels, 1, 'single');
    end    
    if isempty(opts.bias)
        bias = zeros(nchannels, 1, 'single');    
    end
    
    mName = opts.wname;
    bName = opts.bname;
    if isempty(mName)
        mName = iLab_dagnn_setParamNameBNmean(nameLayer); % mean name    
    end    
    if isempty(bName)
        bName = iLab_dagnn_setParamNameBias(nameLayer);   % bias name    
    end
    
    %'name', {sprintf('%sm', nameLayer), sprintf('%sb', nameLayer)}, ...
    block = dagnn.BatchNorm();
    params = struct( ...
                  'name', {mName, bName}, ...
                  'value', {weight, bias}, ...
                  'learningRate', {learningRate(1), learningRate(2)}, ...
                  'weightDecay', {weightDecay(1), weightDecay(2)});
                            
              
    net.addLayer(...
               nameLayer, ...
               block, ...
               inputs, ...
               outputs, ...
               {params.name});
           
    mindex = net.getParamIndex(params(1).name);
    bindex = net.getParamIndex(params(2).name);
    
	net.params(mindex).value        = weight;
    net.params(bindex).value        = bias;
    net.params(mindex).learningRate = learningRate(1);
    net.params(bindex).learningRate = learningRate(2);
    net.params(mindex).weightDecay  = weightDecay(1);
    net.params(bindex).weightDecay  = weightDecay(2); 

end