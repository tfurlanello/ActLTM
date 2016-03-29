function net = iLab_dagnn_getParamUpdationLists(net)
    % which parameters are needed to be updated
    
    if ~isa(net, 'dagnn.DagNN')
        error('wrong type of network architecture\n');
    end
    
    updatelists = [];
    for l=1:numel(net.layers)
        if isa(net.layers(l).block, 'dagnn.Loss') == 0
            updatelists = cat(2, updatelists, net.layers(l).paramIndexes);
        end
    end
    
    net.updatelists = unique(updatelists);

end