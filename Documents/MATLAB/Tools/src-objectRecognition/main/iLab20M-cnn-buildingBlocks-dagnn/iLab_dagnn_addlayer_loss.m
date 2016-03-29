function [net, new_nameLayer, new_outputs] = ...
            iLab_dagnn_addlayer_loss(net, nameLayer, inputs, outputs, opts)
    if ~isa(net, 'dagnn.DagNN')
        error('wrong type of network architecture\n');
    end
    
    opts = iLab_arg2struct(opts);
    opts = iLab_dagnn_validateLossParam(opts);
    
    losstype        = opts.type;
    isstructured    = opts.isstructured;
    labelgraph      = opts.labelgraph;

     params = struct(...
        'name', {sprintf('%sisstructured', nameLayer), sprintf('%slabelgraph', nameLayer)}, ...
        'value', {isstructured, labelgraph});
    
    switch losstype    
        case {'softmaxlog'}
            block = dagnn.Loss('loss', 'softmaxlog') ;    
        case {'crossentropy'}
            block = dagnn.Loss('loss', 'crossentropy');
        case {'classerror-crossentropy'}
            block = dagnn.Loss('loss', 'classerror-crossentropy');
        otherwise
            error('unsupported loss types\n');
    end
 
    
    
    net.addLayer(...
               nameLayer, ...
               block, ...
               inputs, ...
               outputs, ...
               {params.name});
           
           
    sindex = net.getParamIndex(params(1).name);
    gindex = net.getParamIndex(params(2).name);
    
	net.params(sindex).value        = isstructured;
    net.params(gindex).value        = labelgraph;
        
    
    
end