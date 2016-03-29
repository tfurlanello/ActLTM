function net = iLab_arc_dagnn_alexnet(nclasses, args, labelgraph)    
% inputs:
%       nclasses       - # of classes
%       args           - hyperparameters to build the deep network
%       labelgraph     - structured label

    narginchk(1,3); 
    if ~exist('args', 'var') || isempty(args)
        args = {};
    end
    if ~exist('labelgraph', 'var') || isempty(labelgraph) || ~isstruct(labelgraph)
        labelgraph = struct('isstructured',false, ...
                            'labelgraph', []);
    end    
    
    opts.batchNormalization     = false ;    
    opts.conv.weightInitMethod  = 'gaussian';
    opts.conv.scale             = 1.0;
    opts.conv.learningRate      = [1 2];
    opts.conv.weightDecay       = [1 0];
    opts.fc.size                = 1024;
    
    opts.bnorm.learningRate = [2 1];
    opts.bnorm.weightDecay  = [0 0];
    
    opts.norm.param     =   [5 1 0.0001/5 0.75];
    opts.pooling.method =   'max';
    opts.dropout.rate   =   0.5;
    
    opts = vl_argparse(opts, args) ;    
 
 	predictionsNames =  {};
    outputsNames     =  {};
    inputsNames      =  {};

    % build a base deep architecture
    [net, layerNames, mapInputs, mapOutputs] = ...
            iLab_dagnn_alexnet(nclasses, opts);

    inputImgSize      = net.meta.normalization.imageSize;
    varSizesBaseModel = net.getVarSizes({mapInputs(layerNames{1}), [inputImgSize 1]});
    nConv = 5;
    szConvout = cell(1, nConv);
    for c=1:nConv
        varindex = net.getVarIndex(mapOutputs(layerNames{c}));
        szConvout{c} = varSizesBaseModel{varindex};
    end 
    
    varPrediction = mapOutputs(layerNames{end});
    infoPortImage = mapInputs(layerNames{1});
    
    inputsNames      = cat(2, inputsNames, infoPortImage);
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
    
    
    net = iLab_dagnn_getParamUpdationLists(net);
    net.inputsNames = inputsNames;
    net.predictionsNames = predictionsNames;
    net.outputsNames = outputsNames;
                            
end