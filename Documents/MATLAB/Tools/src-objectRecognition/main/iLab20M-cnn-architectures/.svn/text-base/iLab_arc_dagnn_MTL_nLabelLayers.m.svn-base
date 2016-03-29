function net = iLab_arc_dagnn_MTL_nLabelLayers(nTasks, nTaskLabels, taskNames, args, ...
                                                baseModelType, labelgraphs)
% inputs:
%       baseModelType  - 'alexnet', or 'vgg-m' (only support these 2 currently)
%       isstructured   - whether use a structured loss function on
%                       environment-labels

% multi-task learning framework
% it handles any number of tasks
% inputs: 
%           nTasks      - number of tasks
%           nTaskLabels - a vector, with length 'nTasks'
%           TaskNames   - the name of each task


    narginchk(2,5);    
    
    opts.batchNormalization     = false ;    
    opts.conv.weightInitMethod  = 'gaussian';
    opts.conv.scale             = 1.0;
    opts.conv.learningRate      = [1 2];
    opts.conv.weightDecay       = [1 0];
    
    opts.bnorm.learningRate = [2 1];
    opts.bnorm.weightDecay  = [0 0];
    
    opts.norm.param     =   [5 1 0.0001/5 0.75];
    opts.pooling.method =   'max';
    opts.dropout.rate   =   0.5;
    
    opts = vl_argparse(opts, args) ;    
    
    if ~exist('baseModelType', 'var') || isempty(baseModelType)
        baseModelType = 'alexnet';
    end
    
    if ~exist('labelgraphs', 'var') || isempty(labelgraphs)        
        labelgraphs = struct('isstructured', {}, 'labelgraph', {});
        for t=1:nTasks
            labelgraphs(t).isstructured = false;
            labelgraphs(t).labelgraph = [];
        end        
    end
    
    % when taskNames are not given
    if ~exist('taskNames', 'var') || isempty(taskNames)
        taskNames = cell(1, nTasks);
        for t=1:nTasks
            taskNames{t} = ['t' num2str(t)];
        end
    end
        
    
% -------------------------------------------------------------------------
%%                                  build a base deep architecture
% -------------------------------------------------------------------------
    switch baseModelType
        case 'alexnet'
            [net, layerNames, mapInputs, mapOutputs] = ...
                    iLab_dagnn_MTL_alexnet(args);
        case 'vgg-m'
            [net, layerNames, mapInputs, mapOutputs] = ...
                    iLab_dagnn_MTL_vgg_m(args);
        otherwise
            error('unsupported basemodel\n');
    end
    
    inputsNames         =  {};
    outputsNames        =  {};
    predictionsNames    =  {};
        
    infoPortImage = mapInputs(layerNames{1});
    inputsNames = cat(2, inputsNames, infoPortImage);
    
   
% -------------------------------------------------------------------------
%%                                 stack label layers onto the top 
% -------------------------------------------------------------------------
    
	lindex = net.getLayerIndex(layerNames{end});
    params = net.layers(lindex).params; 
    pindex = net.getParamIndex(params{1});
    nNodes = size(net.params(pindex).value,4);
    
 
    for t=1:nTasks    

        varInput    = mapOutputs(layerNames{end});
        nameLayer   = taskNames{t};
        nCategory   = nTaskLabels(t);

        nameLayer   =   iLab_dagnn_getNewLayerName(net, nameLayer);
        inputs      =   varInput;
        outputs     =  sprintf('%sout', nameLayer);

        net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                                    {'size', [1 1 nNodes nCategory],...
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

        if ~labelgraphs(t).isstructured
            % loss layer
            nameLayer       =   iLab_dagnn_getNewLayerName(net, ['loss_' taskNames{t}]);
            varObjective    =   iLab_dagnn_getNewVarName(net,   ['objective_' taskNames{t}]);
            varLabel        =   iLab_dagnn_getNewVarName(net,   ['label_' taskNames{t}]);   

            infoPortLabel = varLabel;

            inputs      = {outputs, varLabel};
            outputs     = varObjective;

            outputsNames = cat(2, outputsNames, varObjective);
            inputsNames = cat(2, inputsNames, infoPortLabel);

            net = iLab_dagnn_addlayer_loss(net,nameLayer, inputs, outputs, ...
                                        {'type', 'softmaxlog', ...
                                        'isstructured', false, ...
                                        'labelgraph', []});
            % error layer
            nameLayer = iLab_dagnn_getNewLayerName(net, ['error_' taskNames{t}]);
            outputs = iLab_dagnn_getNewVarName(net,     ['top1error_' taskNames{t}]);

            net.addLayer( nameLayer, ...
                          dagnn.Loss('loss', 'classerror'), ...
                          inputs, ...
                          outputs);

        else
            % loss layer
            nameLayer       =   iLab_dagnn_getNewLayerName(net, ['loss_' taskNames{t}]);
            varObjective    =   iLab_dagnn_getNewVarName(net,   ['objective_' taskNames{t}]);
            varLabel        =   iLab_dagnn_getNewVarName(net,   ['label_' taskNames{t}]);     

            infoPortLabel = varLabel;

            inputs      = {outputs, varLabel};
            outputs     = varObjective;

            outputsNames = cat(2, outputsNames, varObjective);
            inputsNames = cat(2, inputsNames, infoPortLabel);        

            net = iLab_dagnn_addlayer_loss(net, nameLayer, inputs, outputs, ...
                                        {'type', 'crossentropy', ...
                                        'isstructured', true, ...
                                        'labelgraph',   labelgraphs_env.labelgraph});

            % error layer
            nameLayer = iLab_dagnn_getNewLayerName(net, ['error_' taskNames{t}]);
            outputs = iLab_dagnn_getNewVarName(net,     ['top1error_' taskNames{t}]);

            net = iLab_dagnn_addlayer_loss(net, nameLayer, inputs, outputs, ...
                                        {'type', 'classerror-crossentropy', ...
                                        'isstructured', true, ...
                                        'labelgraph',   labelgraphs(t).labelgraph}); 
        end  

    end

    net = iLab_dagnn_getParamUpdationLists(net);
    net.inputsNames      = inputsNames;
    net.outputsNames     = outputsNames;
    net.predictionsNames = predictionsNames;
      
     
end