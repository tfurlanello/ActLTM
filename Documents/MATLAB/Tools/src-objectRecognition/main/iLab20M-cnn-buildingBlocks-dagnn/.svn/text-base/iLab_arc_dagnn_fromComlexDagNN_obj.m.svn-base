function net =  iLab_arc_dagnn_fromComlexDagNN_obj(trained_net, labelgraph)
% initialize parameters of a linear-chain dagnn  from a trained complex
% dagnn
% input         
%       trained_net - trained deep network
%       labelgraph  - labelgraph


    narginchk(1,2);
    if ~isa(trained_net, 'dagnn.DagNN')
        error('wrong type of architecture, only dagnn architecture are supported\n');
    end
    
    if ~exist('labelgraph', 'var') || isempty(labelgraph)
        labelgraph = struct('isstructured', false, 'labelgraph', []);
    end
    
    
    nLayers = numel(trained_net.layers);
    net = dagnn.DagNN();
    l = 0;
    while 1
        l = l+1;
        if l > nLayers
            break;
        end        
        if isa(trained_net.layers(l).block, 'dagnn.Loss')
            break;
        end
        
        layerName =  trained_net.layers(l).name;
        block     =  trained_net.layers(l).block;
        inputs    =  trained_net.layers(l).inputs;
        outputs   =  trained_net.layers(l).outputs;
        params    =  trained_net.layers(l).params;
        net.addLayer(...
                        layerName, ...
                        block, ...
                        inputs, ...
                        outputs, ...
                        params);
        
        pindex = trained_net.layers(l).paramIndexes;
        p = net.getParamIndex(params);
        assert(numel(pindex) == numel(p));
        
        for j=1:numel(p)
            net.params(p(j)) = trained_net.params(pindex(j));
        end
               
    end
    
    inputsNames         =  {};
    outputsNames        =  {};
    predictionsNames    =  {};
    
    inputsNames = cat(2, inputsNames, net.layers(1).inputs);
    varPrediction = net.layers(end).outputs{1};
    predictionsNames = cat(2, predictionsNames, varPrediction);
    
    % add loss and error layers
    if ~labelgraph.isstructured
        % loss layer
        nameLayer    = iLab_dagnn_getNewLayerName(net, 'loss');
        varObjective = iLab_dagnn_getNewVarName(net, 'objective');
        varLabel     = iLab_dagnn_getNewVarName(net, 'label'); 
        outputsNames = cat(2, outputsNames, varObjective);
        inputsNames  = cat(2, inputsNames, varLabel);
        
        inputs      = {varPrediction, varLabel};
        outputs     = varObjective;

        net = iLab_dagnn_addlayer_loss(net,nameLayer, inputs, outputs, ...
                                    {'type', 'softmaxlog', ...
                                    'isstructured', false, ...
                                    'labelgraph', []});
        % error layer
        nameLayer = iLab_dagnn_getNewLayerName(net, 'error');
        outputs   = iLab_dagnn_getNewVarName(net, 'top1error');
        
        net.addLayer( nameLayer, ...
                      dagnn.Loss('loss', 'classerror'), ...
                      inputs, ...
                      outputs);
        
    else
        % loss layer
        nameLayer    = iLab_dagnn_getNewLayerName(net, 'loss');
        varObjective = iLab_dagnn_getNewVarName(net, 'objective');
        varLabel     = iLab_dagnn_getNewVarName(net, 'label');  
        outputsNames = cat(2, outputsNames, varObjective);
        inputsNames  = cat(2, inputsNames, varLabel);
        
        inputs      = {varPrediction, varLabel};
        outputs     = varObjective;
        
        net = iLab_dagnn_addlayer_loss(net, nameLayer, inputs, outputs, ...
                                    {'type', 'crossentropy', ...
                                    'isstructured', true, ...
                                    'labelgraph',   labelgraph.labelgraph});
                                
        % error layer
        nameLayer = iLab_dagnn_getNewLayerName(net, 'error');
        outputs   = iLab_dagnn_getNewVarName(net, 'top1error');
        
        net = iLab_dagnn_addlayer_loss(net, nameLayer, inputs, outputs, ...
                                    {'type', 'classerror-crossentropy', ...
                                    'isstructured', true, ...
                                    'labelgraph',   labelgraph.labelgraph}); 
    end
    
    net.meta = trained_net.meta;
    net = iLab_dagnn_getParamUpdationLists(net);
    net.inputsNames = inputsNames;
    net.predictionsNames = predictionsNames;
    net.outputsNames = outputsNames;
    
         
    
end