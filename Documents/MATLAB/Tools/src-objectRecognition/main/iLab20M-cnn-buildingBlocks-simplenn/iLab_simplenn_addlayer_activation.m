function net = iLab_simplenn_addlayer_activation(net, name, type)
   
    % type: relu, sigmoid
    
    net.layers{end+1} = struct('type', type, 'name', name) ;


end