function varargout = iLab_dagnn_base_alexnet(args)  
    
    [net, LayerNames, mapInputs, mapOutputs] = ...
                                iLab_dagnn_MTL_alexnet(args);    
    varargout = {net, LayerNames, mapInputs, mapOutputs};

end