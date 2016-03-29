%  this function will return two architectures:
%  1. a simple linear feedforward architecture (e.g. alexnet, vgg) used to do
%     object recognition
%  2. a more complex architecture, builded on the simple architecture, but
%     have skip connection from bottom convolutional layers to the top
%     output layer. The skip connections serve the support of
%     regularization.
%  Note: STL - means "single task learning", as opposed to Multi-task
%              learning
%        in the future, we should design a more generalized construction
%        function, which support any kinds of skip connections. Currently,
%        we have to hard-code the layer index, e.g., 1,2,3,4 in the function 
%        name, which is not flexible.

function [net, netLinear] = iLab_arc_dagnn_STL_multiLevelInjection_conv1234fc2(nclasses_obj, ...
                                                            args, baseModelType)
% inputs:
%       nclasses_obj   - the # of object classes
%       nclasses_env   - the # of environment classes
%       baseModelType  - 'alexnet', or 'vgg-m' (only support these 2 currently)
%       isstructured   - whether use a structured loss function on
%                       environment-labels

    narginchk(1,3);    
    
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
    
    
    predictionsNames = {};
    outputsNames = {};
    % build a base deep architecture
    switch baseModelType
        case 'alexnet'
            [net, layerNames, mapInputs, mapOutputs] = ...
                    iLab_dagnn_alexnet(nclasses_obj, opts);
        case 'vgg-m'
            [net, layerNames, mapInputs, mapOutputs] = ...
                    iLab_dagnn_vgg_m(nclasses_obj, opts);
        otherwise
            error('unsupported basemodel\n');
    end

    inputImgSize      = net.meta.normalization.imageSize;
    varSizesBaseModel = net.getVarSizes({mapInputs(layerNames{1}), [inputImgSize 1]});
    nConv = 5;
    szConvout = cell(1, nConv);
    for c=1:nConv
        varindex = net.getVarIndex(mapOutputs(layerNames{c}));
        szConvout{c} = varSizesBaseModel{varindex};
    end    
    
    infoPortImage = mapInputs(layerNames{1});
    
    % add "object prediction" loss & error layer
    varPred = mapOutputs(layerNames{end});
    nameLayer       = 'loss_obj';
    varLabelObj     = 'label_obj';
    varObjectiveObj = 'objective_obj';
    
    infoPortLabelObj = varLabelObj;
    
    nameLayer   =   iLab_dagnn_getNewLayerName(net, nameLayer);
    varLabelObj     = iLab_dagnn_getNewVarName(net, varLabelObj);
    varObjectiveObj = iLab_dagnn_getNewVarName(net, varObjectiveObj);
    outputsNames = cat(2, outputsNames, varObjectiveObj);
    
    inputs      =   {varPred, varLabelObj};
    outputs     =   varObjectiveObj;
    predictionsNames = cat(2, predictionsNames, varPred);
    
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
    
% 	lindex = net.getLayerIndex(layerNames{end});
%     params = net.layers(lindex).params; 
%     pindex = net.getParamIndex(params{1});
%     nNodes = size(net.params(pindex).value,3);

    
    %% output a linear chain deep architecture
%     netLinear = dagnn.DagNN.setobj(net);
%     netLinear = iLab_dagnn_getParamUpdationLists(netLinear);
%     netLinear.inputsNames       = {infoPortImage, infoPortLabelObj};
%     netLinear.outputsNames      = {varObjectiveObj};
%     netLinear.predictionsNames  = {varPred};
      netLinear = [];
      
    
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% until now, we build a linear-structured deep architecture, used 
% to do object recognition
% then, we add skip connections from convolutional layers to the output
% layer
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    

    
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
% start: inject environment information into the earlier
% convolutional layers
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    varPredObj = varPred;
    
    fc1size = 512;
    fc2size = 512;
    %% inject into conv1
     
    inputs           =  mapOutputs(layerNames{1});
    inputSize        =  szConvout{1};
    outputSize1      =  fc1size;
    layerNamePrefix1 =  'conv11';
    [net, outputs] = iLab_dagnn_addlayer_fc(net, inputSize, outputSize1, ...
                                                    layerNamePrefix1, inputs, opts); 	
    
    inputs           =  outputs;
    inputSize        =  [1 1 outputSize1 1];
    outputSize2      =  fc2size;
    layerNamePrefix2 =  'conv12';
    [net, outputs] = iLab_dagnn_addlayer_fc(net, inputSize, outputSize2, ...
                                                    layerNamePrefix2, inputs, opts); 	
    
    nameLayer   =   'conv1Prediction_obj';
    inputs      =   outputs;
    outputs     =   varPredObj;
    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                            {'size', [1 1 outputSize2 nclasses_obj],...
                            'stride', 1, 'pad', 0, ...
                            'learningRate', opts.conv.learningRate, ...
                            'weightDecay',  opts.conv.weightDecay, ...
                            'weightInitMethod', opts.conv.weightInitMethod, ...
                            'scale', opts.conv.scale});
    
    
    %% inject into conv2
	inputs           =  mapOutputs(layerNames{2});
    inputSize        =  szConvout{2};
    outputSize1      =  fc1size;
    layerNamePrefix1 =  'conv21';
    [net, outputs] = iLab_dagnn_addlayer_fc(net, inputSize, outputSize1, ...
                                                    layerNamePrefix1, inputs, opts); 	
    
    inputs           =  outputs;
    inputSize        =  [1 1 outputSize1 1];
    outputSize2      =  fc2size;
    layerNamePrefix2 =  'conv22';
    [net, outputs] = iLab_dagnn_addlayer_fc(net, inputSize, outputSize2, ...
                                                    layerNamePrefix2, inputs, opts); 	
    
    nameLayer   =   'conv2Prediction_obj';
    inputs      =   outputs;
    outputs     =   varPredObj;
    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                            {'size', [1 1 outputSize2 nclasses_obj],...
                            'stride', 1, 'pad', 0, ...
                            'learningRate', opts.conv.learningRate, ...
                            'weightDecay',  opts.conv.weightDecay, ...
                            'weightInitMethod', opts.conv.weightInitMethod, ...
                            'scale', opts.conv.scale});  
    
    
    %}
    %% inject into conv3
    fc1size = 1024;
    fc2size = 1024;    
    inputs           =  mapOutputs(layerNames{3});
    inputSize        =  szConvout{3};
    outputSize1      =  fc1size;
    layerNamePrefix1 =  'conv31';
    [net, outputs] = iLab_dagnn_addlayer_fc(net, inputSize, outputSize1, ...
                                                    layerNamePrefix1, inputs, opts); 	
    
    inputs           =  outputs;
    inputSize        =  [1 1 outputSize1 1];
    outputSize2      =  fc2size;
    layerNamePrefix2 =  'conv32';
    [net, outputs] = iLab_dagnn_addlayer_fc(net, inputSize, outputSize2, ...
                                                    layerNamePrefix2, inputs, opts); 	
    
    nameLayer   =   'conv3Prediction_obj';
    inputs      =   outputs;
    outputs     =   varPredObj;
    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                            {'size', [1 1 outputSize2 nclasses_obj],...
                            'stride', 1, 'pad', 0, ...
                            'learningRate', opts.conv.learningRate, ...
                            'weightDecay',  opts.conv.weightDecay, ...
                            'weightInitMethod', opts.conv.weightInitMethod, ...
                            'scale', opts.conv.scale});    
    
    
    
    %% inject into conv4
	inputs           =  mapOutputs(layerNames{4});
    inputSize        =  szConvout{4};
    outputSize1      =  fc1size;
    layerNamePrefix1 =  'conv41';
    [net, outputs] = iLab_dagnn_addlayer_fc(net, inputSize, outputSize1, ...
                                                    layerNamePrefix1, inputs, opts); 	
    
    inputs           =  outputs;
    inputSize        =  [1 1 outputSize1 1];
    outputSize2      =  fc2size;
    layerNamePrefix2 =  'conv42';
    [net, outputs] = iLab_dagnn_addlayer_fc(net, inputSize, outputSize2, ...
                                                    layerNamePrefix2, inputs, opts); 	
    
    nameLayer   =   'conv4Prediction_obj';
    inputs      =   outputs;
    outputs     =   varPredObj;
    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                            {'size', [1 1 outputSize2 nclasses_obj],...
                            'stride', 1, 'pad', 0, ...
                            'learningRate', opts.conv.learningRate, ...
                            'weightDecay',  opts.conv.weightDecay, ...
                            'weightInitMethod', opts.conv.weightInitMethod, ...
                            'scale', opts.conv.scale});   
                        
    %{
    %% inject into conv5
	inputs           =  mapOutputs(layerNames{5});
    inputSize        =  szConvout{5};
    outputSize1      =  fc1size;
    layerNamePrefix1 =  'conv51';
    [net, outputs] = iLab_dagnn_addlayer_fc(net, inputSize, outputSize1, ...
                                                    layerNamePrefix1, inputs, opts); 	
    
    inputs           =  outputs;
    inputSize        =  [1 1 outputSize1 1];
    outputSize2      =  fc2size;
    layerNamePrefix2 =  'conv52';
    [net, outputs] = iLab_dagnn_addlayer_fc(net, inputSize, outputSize2, ...
                                                    layerNamePrefix2, inputs, opts); 	
    
    nameLayer   =   'conv5Prediction_env';
    inputs      =   outputs;
    outputs     =   varPredictionEnv;
    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                            {'size', [1 1 outputSize2 nclasses_env],...
                            'stride', 1, 'pad', 0, ...
                            'learningRate', opts.conv.learningRate, ...
                            'weightDecay',  opts.conv.weightDecay, ...
                            'weightInitMethod', opts.conv.weightInitMethod, ...
                            'scale', opts.conv.scale});                           
   %} 
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
% end: inject environment information into the earlier
% convolutional layers
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX   
    
    
    
    % which parameters are needed to be updated
%     updatelists = [];
%     for l=1:numel(net.layers)
%         if isa(net.layers(l).block, 'dagnn.Loss') == 0
%             updatelists = cat(2, updatelists, net.layers(l).paramIndexes);
%         end
%     end
%     net.updatelists = updatelists;
%     
    net = iLab_dagnn_getParamUpdationLists(net);
    net.inputsNames = {infoPortImage, infoPortLabelObj};
    net.predictionsNames = predictionsNames;
    net.outputsNames = outputsNames;
     
     
end