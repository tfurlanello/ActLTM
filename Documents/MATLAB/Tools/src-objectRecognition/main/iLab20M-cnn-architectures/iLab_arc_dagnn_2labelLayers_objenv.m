function [net, netLinear] = iLab_arc_dagnn_2labelLayers_objenv(nclasses_obj, nclasses_env, args, ...
                                                baseModelType, labelgraph_env)
% inputs:
%       nclasses_obj   - the # of object classes
%       nclasses_env   - the # of environment classes
%       baseModelType  - 'alexnet', or 'vgg-m' (only support these 2 currently)
%       isstructured   - whether use a structured loss function on
%                       environment-labels

    narginchk(3,5);    
    
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
    
    if ~exist('baseModelType', 'var') || isempty(baseModelType)
        baseModelType = 'alexnet';
    end
    
    if ~exist('labelgraph_env', 'var') || isempty(labelgraph_env)
        labelgraph_env = struct('isstructured', false, 'labelgraph', []);
    end
    
    % build a base deep architecture
    switch baseModelType
        case 'alexnet'
            [net, layerNames, mapInputs, mapOutputs] = ...
                    iLab_dagnn_alexnet(nclasses_obj, args);
        case 'vgg-m'
            [net, layerNames, mapInputs, mapOutputs] = ...
                    iLab_dagnn_vgg_m(nclasses_obj, args);
        otherwise
            error('unsupported basemodel\n');
    end
    
    inputsNames      = {};
    outputsNames     = {};
    predictionsNames = {};
        
    infoPortImage = mapInputs(layerNames{1});
    inputsNames = cat(2, inputsNames, infoPortImage);
    
    % add "object prediction" loss & error layer
    varPred = mapOutputs(layerNames{end});
    nameLayer   =   'loss_obj';
    varLabelObj     = 'label_obj';
    varObjectiveObj = 'objective_obj';
    
    infoPortLabelObj = varLabelObj;
    
    nameLayer   =   iLab_dagnn_getNewLayerName(net, nameLayer);
    varLabelObj     = iLab_dagnn_getNewVarName(net, varLabelObj);
    varObjectiveObj = iLab_dagnn_getNewVarName(net, varObjectiveObj);
    
    inputs      =   {varPred, varLabelObj};
    outputs     =   varObjectiveObj;
    
    predictionsNames = cat(2, predictionsNames, varPred);
    outputsNames     = cat(2, outputsNames, varObjectiveObj);
    inputsNames = cat(2, inputsNames, infoPortLabelObj);
    
    net = iLab_dagnn_addlayer_loss(net,nameLayer, inputs, outputs, ...
                                {'type', 'softmaxlog', ...
                                'isstructured', false, ...
                                'labelgraph', []});
    
    nameLayer   =   'error_obj';
    varErrorObj =   'top1error_obj';    
    
    nameLayer   = iLab_dagnn_getNewLayerName(net, nameLayer);
    varErrorObj = iLab_dagnn_getNewVarName(net, varErrorObj);
    
    inputs      =   {varPred, varLabelObj};
    outputs     =   varErrorObj;
    
    net.addLayer(nameLayer, ...
                 dagnn.Loss('loss', 'classerror'), ...
                 inputs, ...
                 outputs);
    
	lindex = net.getLayerIndex(layerNames{end});
    params = net.layers(lindex).params; 
    pindex = net.getParamIndex(params{1});
    nNodes = size(net.params(pindex).value,3);
    
    %% output a linear chain deep architecture
%     netLinear = dagnn.DagNN.setobj(net);
%     netLinear = iLab_dagnn_getParamUpdationLists(netLinear);
%     netLinear.inputsNames       = {infoPortImage, infoPortLabelObj};
%     netLinear.outputsNames      = {varObjectiveObj};
%     netLinear.predictionsNames  = {varPred};    
    netLinear = [];

%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
% until now, we have finished creating a classic dagnn architecture
% now, inject environment labels into the top layer
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX    
    % add "picture-taken-environment" prediction, loss and error layer 
    % prediction layer
    varInput    = mapInputs(layerNames{end});
    nameLayer   = 'prediction_env';
    
    nameLayer   =   iLab_dagnn_getNewLayerName(net, nameLayer);
    inputs      =   varInput;
    outputs     =  sprintf('%sout', nameLayer);

    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                                {'size', [1 1 nNodes nclasses_env],...
                                'stride', 1, 'pad', 0, ...
                                'learningRate', opts.conv.learningRate, ...
                                'weightDecay',  opts.conv.weightDecay, ...
                                'weightInitMethod', opts.conv.weightInitMethod, ...
                                'scale', opts.conv.scale});
   
	predictionsNames = cat(2, predictionsNames, outputs);
                            
    switch lower(opts.conv.weightInitMethod)
      case {'xavier', 'xavierimproved'}
        lindex = net.getLayerIndex(nameLayer);
        params = net.layers(lindex).params;
        pindex = net.getParamIndex(params{1});
        net.params(pindex).value = net.params(pindex).value / 10; 
    end     
    
    if ~labelgraph_env.isstructured
        % loss layer
        nameLayer       = iLab_dagnn_getNewLayerName(net, 'loss_env');
        varObjectiveEnv = iLab_dagnn_getNewVarName(net, 'objective_env');
        varLabelEnv     = iLab_dagnn_getNewVarName(net, 'label_env');   
        
        infoPortLabelEnv = varLabelEnv;
        
        inputs      = {outputs, varLabelEnv};
        outputs     = varObjectiveEnv;
        
        outputsNames = cat(2, outputsNames, varObjectiveEnv);
        inputsNames = cat(2, inputsNames, infoPortLabelEnv);

        net = iLab_dagnn_addlayer_loss(net,nameLayer, inputs, outputs, ...
                                    {'type', 'softmaxlog', ...
                                    'isstructured', false, ...
                                    'labelgraph', []});
        % error layer
        nameLayer = iLab_dagnn_getNewLayerName(net, 'error_env');
        outputs = iLab_dagnn_getNewVarName(net, 'top1error_env');
        
        net.addLayer( nameLayer, ...
                      dagnn.Loss('loss', 'classerror'), ...
                      inputs, ...
                      outputs);
        
    else
        % loss layer
        nameLayer       = iLab_dagnn_getNewLayerName(net, 'loss_env');
        varObjectiveEnv = iLab_dagnn_getNewVarName(net, 'objective_env');
        varLabelEnv     = iLab_dagnn_getNewVarName(net, 'label_env');      
        
        infoPortLabelEnv = varLabelEnv;
        
        inputs      = {outputs, varLabelEnv};
        outputs     = varObjectiveEnv;
        
        outputsNames = cat(2, outputsNames, varObjectiveEnv);
        inputsNames = cat(2, inputsNames, infoPortLabelEnv);        
        
        net = iLab_dagnn_addlayer_loss(net, nameLayer, inputs, outputs, ...
                                    {'type', 'crossentropy', ...
                                    'isstructured', true, ...
                                    'labelgraph',   labelgraph_env.labelgraph});
                                
        % error layer
        nameLayer = iLab_dagnn_getNewLayerName(net, 'error_env');
        outputs   = iLab_dagnn_getNewVarName(net, 'top1error_env');
        
        net = iLab_dagnn_addlayer_loss(net, nameLayer, inputs, outputs, ...
                                    {'type', 'classerror-crossentropy', ...
                                    'isstructured', true, ...
                                    'labelgraph',   labelgraph_env.labelgraph}); 
    end  
    
    % which parameters are needed to be updated
%     updatelists = [];
%     for l=1:numel(net.layers)
%         if isa(net.layers(l).block, 'dagnn.Loss') == 0
%             updatelists = cat(2, updatelists, net.layers(l).paramIndexes);
%         end
%     end
%     net.updatelists = updatelists;

    net = iLab_dagnn_getParamUpdationLists(net);
    net.inputsNames      = inputsNames;
    net.outputsNames     = outputsNames;
    net.predictionsNames = predictionsNames;
      
     
end