function varargout = iLab_dagnn_base_vgg_m(args)  
    
    [net, LayerNames, mapInputs, mapOutputs] = ...
                                iLab_dagnn_MTL_vgg_m(args);    
    varargout = {net, LayerNames, mapInputs, mapOutputs};

end