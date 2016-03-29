function net = iLab_arc_de_dagnn_2streams_woL2( ...
                            nclasses, ntransformations, param_init_net, ...
                            bshare_s, bshare_lf, nfactors_lf, fractions_lf, args, labelgraph)    
% functionality:
% it is desgined to disentangle the instantiation factors from the object
% identity, such that the object recognition is independent of the
% environmental factors, such as lighting conditions, camera view points ...

% please make a reference to the inverse graphics machine by "joshua
% Tenembaum", MIT

% inputs:
%       nclasses       - # of object classes
%       ntranformation - # of environmental classes
%       param_init_net - an architecture, which initializes parameters for 
%                        different target neural networks
%       bshare_s       - wheather two streams of convnets share the same
%                        parameter or not
%       bshare_lf      - wheather latent factors on top of different streams 
%                        share the same parameters 
%       args           - hyperparameters to build the deep network
%       labelgraph     - structured label

%% notes:
% this prototyping script is more elegant than
% 'iLab_arc_dagnn_MTL_2streams_disentangling.m': although they are
% equivalent mathematically, but the former one has fewer layers.
% Specifically, instead of building fc7 first, and then split it into
% identity and transformation units, in this script, fc6 is directly
% splitted into identity and transformation units!

%% additional notes:
% This architecture has NO 'l2-loss' layer.

    narginchk(2,9); 
    if ~exist('args', 'var') || isempty(args)
        args = {};
    end
    if ~exist('labelgraph', 'var') || isempty(labelgraph) || ~isstruct(labelgraph)
        labelgraph = struct('isstructured',false, ...
                            'labelgraph', []);
    end    
    if ~exist('bshare_s' ,'var') || isempty(bshare_s)
        bshare_s  = true;
    end    
    if ~exist('bshare_lf', 'var') || isempty(bshare_lf)
        bshare_lf = true;
    end
    
    if ~exist('nfactors_lf', 'var') || isempty(nfactors_lf)
        nfactors_lf = 2;
    end
    
    if ~exist('fractions_lf', 'var') || isempty(fractions_lf)
        fractions_lf = ones(1,nfactors_lf) * 1/nfactors_lf;
    end    
    
    if labelgraph.isstructured
        error('This architecture doesn''t support structured output\n');
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
 
    nstreams = 2;
    nfactors = nfactors_lf;
    nfractions = fractions_lf;
    

   %% build a deep base architecture
    % build a base deep architecture, using either alexnet or vgg-m
    % the base consists of 2 parallel streams, which could share or use
    % different convolutional parameters
    % On the top layer, fc6, we further stack different latent factors onto
    % the top.
    [net, layersNames, mapsInputs, mapsOutputs, ...
        lf_layersNames, lf_mapsInputs, lf_mapsOutputs] = ...
                iLab_dagnn_streams_alexnet6_factors(nstreams, bshare_s, ...
                                            nfactors, nfractions, bshare_lf, opts); 
    % check: 2 streams, and each stream has two latent factors on the top
    assert(numel(layersNames) == nstreams);
    assert(numel(mapsInputs)  == nstreams);
    assert(numel(mapsOutputs) == nstreams);
    assert(numel(lf_layersNames) == nfactors);
    assert(numel(lf_mapsInputs)  == nfactors);
    assert(numel(lf_mapsOutputs) == nfactors);
    net.mapsInputs      =  mapsInputs;
    net.mapsOutputs     =  mapsOutputs;
    net.layersNames     =  layersNames;
    net.lf_mapsInputs   =  lf_mapsInputs;
    net.lf_mapsOutputs  =  lf_mapsOutputs;
    net.lf_layersNames  =  lf_layersNames;
    
    
    %% get the size of convolutional layers in the base architecture
    inputImgSize = net.meta.normalization.imageSize;
    mapsInputs1  = mapsInputs{1}; 
    layersNames1 = layersNames{1};
    mapsOutputs1 = mapsOutputs{1};
    mapsInputs2  = mapsInputs{2};
    layersNames2 = layersNames{2};    
    mapsOutputs2 = mapsOutputs{2};
    varSizesBaseModel =  net.getVarSizes({mapsInputs1(layersNames1{1}), [inputImgSize 1], ...
                                   mapsInputs2(layersNames2{1}), [inputImgSize 1]});
    nConv = numel(layersNames1);
    szConvout = cell(1, nConv);
    for c=1:nConv
        varindex = net.getVarIndex(mapsOutputs1(layersNames1{c}));
        szConvout{c} = varSizesBaseModel{varindex};
    end     
    szTopLayer = szConvout{end};
    outputsize = szTopLayer;
    
    assert(outputsize(1) == 1); 
    assert(outputsize(2) == 1); 
    assert(outputsize(4) == 1);   
    
    %% get the size of latent layers (on the top of fc6)
    szLatentFactorsOut = cell(nstreams, nfactors);
    for s=1:nstreams
        s_lf_layersNames = lf_layersNames{s};
        s_lf_mapsOutputs = lf_mapsOutputs{s};
        for lf=1:nfactors
            varindex = net.getVarIndex(s_lf_mapsOutputs(s_lf_layersNames{lf}));
            szLatentFactorsOut{s,lf} = varSizesBaseModel{varindex};
        end
    end
 
    assert(szLatentFactorsOut{1,1}(3) == szLatentFactorsOut{2,1}(3));
    assert(szLatentFactorsOut{1,2}(3) == szLatentFactorsOut{2,2}(3));
    
    nIdentity       = szLatentFactorsOut{1,1}(3);
    nTransformation = szLatentFactorsOut{1,2}(3);
    
    representationsIdentity = {lf_mapsOutputs{1}(lf_layersNames{1}{1}), ...
                                  lf_mapsOutputs{2}(lf_layersNames{2}{1})};
    
    representationsTransform = {lf_mapsOutputs{1}(lf_layersNames{1}{2}), ...
                                  lf_mapsOutputs{2}(lf_layersNames{2}{2})};
                              
    inputsNames = cat(2, inputsNames, mapsInputs1(layersNames1{1}));
    inputsNames = cat(2, inputsNames, mapsInputs2(layersNames2{1}));    

    
% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                    add identity and transformation layers
% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX   
   %% identity prediction1
	nameLayer = 'predictionIdentity';
    inputs = representationsIdentity{1};
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
    outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));
    
    predictionsIdentity = outputs;
    
    predictionsNames = cat(2, predictionsNames, predictionsIdentity);
    
	fname = iLab_dagnn_setParamNameFilter(nameLayer);
    bname = iLab_dagnn_setParamNameBias(nameLayer); 
    
    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                {'size', [1 1 nIdentity nclasses], 'stride', 1, 'pad', 0, ...
                'filter',           [], ...
                'bias',             [], ...
                'fname',            fname, ...
                'bname',            bname, ...
                'learningRate',     opts.conv.learningRate, ...
                'weightDecay',      opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale',            opts.conv.scale});
    
    % final touches
    switch lower(opts.conv.weightInitMethod)
      case {'xavier', 'xavierimproved'}
        lindex = net.getLayerIndex(nameLayer);
        params = net.layers(lindex).params;
        pindex = net.getParamIndex(params{1});
        net.params(pindex).value = net.params(pindex).value / 10;  
        
    end  
            
    %% transformation prediction1 
	nameLayer = 'predictionTransform';    
    inputs = representationsTransform{1};
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
    outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));
    
    predictionsTransform = outputs;
    
	fname = iLab_dagnn_setParamNameFilter(nameLayer);
    bname = iLab_dagnn_setParamNameBias(nameLayer); 
    
    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                {'size', [1 1 nTransformation ntransformations], 'stride', 1, 'pad', 0, ...
                'filter',           [], ...
                'bias',             [], ...
                'fname',            fname, ...
                'bname',            bname, ...
                'learningRate',     opts.conv.learningRate, ...
                'weightDecay',      opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale',            opts.conv.scale});
    
    % final touches
    switch lower(opts.conv.weightInitMethod)
      case {'xavier', 'xavierimproved'}
        lindex = net.getLayerIndex(nameLayer);
        params = net.layers(lindex).params;
        pindex = net.getParamIndex(params{1});
        net.params(pindex).value = net.params(pindex).value / 10;   
        
    end
           
 
    %% transformation prediction2         
	nameLayer = 'predictionTransform2';
    inputs = representationsTransform{2};
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
    outputs   = predictionsTransform;
    
	predictionsNames = cat(2, predictionsNames, predictionsTransform);

    
	fname = iLab_dagnn_setParamNameFilter(nameLayer);
    bname = iLab_dagnn_setParamNameBias(nameLayer); 
    
    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                {'size', [1 1 nTransformation ntransformations], 'stride', 1, 'pad', 0, ...
                'filter',           [], ...
                'bias',             [], ...
                'fname',            fname, ...
                'bname',            bname, ...
                'learningRate',     opts.conv.learningRate, ...
                'weightDecay',      opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale',            opts.conv.scale});
    
    % final touches
    switch lower(opts.conv.weightInitMethod)
      case {'xavier', 'xavierimproved'}
        lindex = net.getLayerIndex(nameLayer);
        params = net.layers(lindex).params;
        pindex = net.getParamIndex(params{1});
        net.params(pindex).value = net.params(pindex).value / 10;  
        
    end  
  
    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %                                          build loss layers to the top
    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX        
    %% add one l2 loss layer, using identity representations from both images
    %% note: the identity representation of the left image is used
    % to regularize the identity representation of the right image
% 	nameLayer    = 'lossL2l';
%     varObjective = 'objectiveL2l';    
%     nameLayer    = iLab_dagnn_getNewLayerName(net, nameLayer);
% 	  varObjective = iLab_dagnn_getNewVarName(net, varObjective);
%     
%     inputs = {representationsIdentity{1}, representationsIdentity{2}};
%     outputs   =  varObjective;    
%     net.addLayer( nameLayer, ...
%                   dagnn.Loss('loss', 'l2'), ...
%                   inputs, ...
%                   outputs);
%               
%     outputsNames = cat(2, outputsNames, varObjective);    
    %% right image identity loss
%     nameLayer    = 'lossL2r';
%     varObjective = 'objectiveL2r';    
%     nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
% 	  varObjective = iLab_dagnn_getNewVarName(net, varObjective);
%     
%     inputs = {representationsIdentity{2}, representationsIdentity{1}};
%     outputs   =  varObjective;    
%     net.addLayer( nameLayer, ...
%                   dagnn.Loss('loss', 'l2'), ...
%                   inputs, ...
%                   outputs);
%     outputsNames = cat(2, outputsNames, varObjective);
    
    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %                                             add loss and error layers
    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX    
    %% object category loss
    nameLayer     =  'lossObject';
    varObjective  =  'objectiveObject';
    varLabel      =  'labelObject';
    nameLayer    = iLab_dagnn_getNewLayerName(net, nameLayer);
    varObjective = iLab_dagnn_getNewVarName(net, varObjective);
    varLabel     = iLab_dagnn_getNewVarName(net, varLabel); 
    
    inputs      = {predictionsIdentity, varLabel};
    outputs     = varObjective;

    net = iLab_dagnn_addlayer_loss(net,nameLayer, inputs, outputs, ...
                                {'type', 'softmaxlog', ...
                                'isstructured', false, ...
                                'labelgraph', []});
    outputsNames = cat(2, outputsNames, varObjective);
                            
    % error layer
    nameLayer = 'errorObject';    
    outputs   = 'top1errorObject';
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
    outputs   = iLab_dagnn_getNewVarName(net, outputs);

    net.addLayer( nameLayer, ...
                  dagnn.Loss('loss', 'classerror'), ...
                  inputs, ...
                  outputs);
        
     inputsNames = cat(2, inputsNames, varLabel);
    %% transformation loss    
    nameLayer     =  'lossTransformation';
    varObjective  =  'objectiveTransformation';
    varLabel      =  'labelTransformation';
    nameLayer    = iLab_dagnn_getNewLayerName(net, nameLayer);
    varObjective = iLab_dagnn_getNewVarName(net, varObjective);
    varLabel     = iLab_dagnn_getNewVarName(net, varLabel);     

    inputs      = {predictionsTransform, varLabel};
    outputs     = varObjective;

    net = iLab_dagnn_addlayer_loss(net,nameLayer, inputs, outputs, ...
                                {'type', 'softmaxlog', ...
                                'isstructured', false, ...
                                'labelgraph', []});
    outputsNames = cat(2, outputsNames, varObjective);
                            
    % error layer
    nameLayer = 'errorTransformation';    
    outputs   = 'top1errorTransformation';
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
    outputs   = iLab_dagnn_getNewVarName(net, outputs);

    net.addLayer( nameLayer, ...
                  dagnn.Loss('loss', 'classerror'), ...
                  inputs, ...
                  outputs);   
              
    inputsNames = cat(2, inputsNames, varLabel);
    
    %% get the updatelists
    net = iLab_dagnn_getParamUpdationLists(net);   
    
    net.inputsNames = inputsNames;
    net.predictionsNames = predictionsNames;
    net.outputsNames = outputsNames;
    
    %% update synchronization lists: (layerlist, size, rate)
    % compute the size of the dropout layer
    % its size is equal to the size the input variable
    if ~isempty(net.synLayers)
        varSizes =  net.getVarSizes({mapsInputs1(layersNames1{1}), [inputImgSize 1], ...
                               mapsInputs2(layersNames2{1}), [inputImgSize 1]});
        nsyn = size(net.synLayers,1);
        % (1)layer lists; (2)layer size; (3) dropout rate
        synLayerInfo = cell(nsyn,4); 
        
        for s=1:nsyn
           ls = net.synLayers{s};           
           for l=1:numel(ls)
                varIdx = net.layers(ls(l)).inputIndexes;
                if l == 1
                    sz = varSizes{varIdx};
                    assert(isa(net.layers(ls(l)).block, 'dagnn.DropOut'));
                    rate = net.layers(ls(l)).block.rate;
                    scale = single(1 / (1 - rate));
                    net.layers(ls(l)).block.frozen = true;
                else
                    sz_ = varSizes{varIdx};
                    assert(isa(net.layers(ls(l)).block, 'dagnn.DropOut'));
                    rate_ = net.layers(ls(l)).block.rate;
                    scale_ = single(1 / (1 - rate_));
                    assert(isequal(sz, sz_));
                    assert(scale_ == scale);
                    net.layers(ls(l)).block.frozen = true;

                end
           end           
           synLayerInfo{s,1} = ls;
           synLayerInfo{s,2} = sz;
           synLayerInfo{s,3} = single(rate);
           synLayerInfo{s,4} = single(scale);            
        end
    end
    net.synLayers = synLayerInfo;
    
    %% reset the parameters by the parameters from param_init_net    
    % notes: the 'param_init_net' should be created by the same
    % architecture-building function, such that parameters are named by the
    % same conventions
    if exist('param_init_net', 'var') && isa(param_init_net, 'dagnn.DagNN')
        tarParamsNames = {net.params.name};
        ntarparams = numel(tarParamsNames);

        for p=1:ntarparams
            paramidx = param_init_net.getParamIndex(tarParamsNames{p});
            if isnan(paramidx) || isempty(paramidx)
                continue;
            end
            tarParam = net.params(p).value;
            refParam = param_init_net.params(paramidx).value;
            if isequal(size(tarParam), size(refParam))
                net.params(p).value = refParam;
            end
        end
        
    end
                         
end