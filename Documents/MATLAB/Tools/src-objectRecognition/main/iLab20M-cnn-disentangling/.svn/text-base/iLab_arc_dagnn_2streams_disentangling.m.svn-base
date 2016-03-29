function net = iLab_arc_dagnn_2streams_disentangling( ...
                            nclasses, ntransformations, bshare, args, labelgraph)    
% functionality:
% it is desgined to disentangle the instantiation factors from the object
% identity, such that the object recognition is independent of the
% environmental factors, such as lighting conditions, camera view points ...

% inputs:
%       nclasses       - # of object classes
%       nEvn           - # of environmental classes
%       bshare         - wheather two streams of convnets share the same
%                        parameter or not
%       args           - hyperparameters to build the deep network
%       labelgraph     - structured label

    narginchk(2,5); 
    if ~exist('args', 'var') || isempty(args)
        args = {};
    end
    if ~exist('labelgraph', 'var') || isempty(labelgraph) || ~isstruct(labelgraph)
        labelgraph = struct('isstructured',false, ...
                            'labelgraph', []);
    end    
    if ~exist('bshare' ,'var') || isempty(bshare)
        bshare = true;
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
    noUpdateLists = [];

    nstreams = 2;
    nfactors = 2;
    nfractions = [1/2 1/2];
    %% build a deep base architecture
    % build a base deep architecture, using either alexnet or vgg-m
    % the base consists of 2 parallel streams, which could share or use
    % different convolutional parameters
    [net, layersNames, mapsInputs, mapsOutputs] = ...
                iLab_dagnn_streams_alexnet(nstreams, bshare, opts); 
    assert(numel(layersNames) == 2);
    assert(numel(mapsInputs)  == 2);
    assert(numel(mapsOutputs) == 2);

    %% get the size of all layers
    inputImgSize = net.meta.normalization.imageSize;
    mapsInputs1  = mapsInputs{1}; 
    layersNames1 = layersNames{1};
    mapsOutputs1 = mapsOutputs{1};
    mapsInputs2  = mapsInputs{2};
    layersNames2 = layersNames{2};    
    mapsOutputs2 = mapsOutputs{2};
    varSizesBaseModel =  net.getVarSizes({mapsInputs1(layersNames1{1}), [inputImgSize 1]});
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
    
    nIdentity       = round(outputsize(3)*nfractions(1));
    nTransformation = outputsize(3) - nIdentity;
    
    inputsNames = cat(2, inputsNames, mapsInputs1(layersNames1{1}));
    inputsNames = cat(2, inputsNames, mapsInputs2(layersNames2{1}));    
    % XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %                                     split the top convolutional layer 
    %                                     into two sub-layers
    %                                     identity and transformation layer
    % XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    representationsIdentity  = {};
    representationsTransform = {};
   %% identity 1
    nameLayer = 'identity1';
    inputs = mapsOutputs1(layersNames1{end});
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
    outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));
    
    representationsIdentity = cat(2, representationsIdentity, outputs);
    
    fname = iLab_dagnn_setParamNameFilter(nameLayer);
    bname = iLab_dagnn_setParamNameBias(nameLayer); 
    in = outputsize(3);
    out = nIdentity;
	bias = zeros(out,1, 'single');
    filter = [diag(ones(1,out, 'single'));  zeros(in-out, out, 'single')]; 
    filter = reshape(filter, [1 1 in out]);

    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                {'size', [1 1 in out], 'stride', 1, 'pad', 0, ...
                'filter',           filter, ...
                'bias',             bias, ...
                'fname',            fname, ...
                'bname',            bname, ...
                'learningRate',     opts.conv.learningRate, ...
                'weightDecay',      opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale',            opts.conv.scale});
    
    lindex        =  net.getLayerIndex(nameLayer);       
    pindex        =  net.layers(lindex).paramIndexes;
    noUpdateLists =  cat(1, noUpdateLists, pindex(:));
            
    %% identity prediction1
	nameLayer = 'predictionIdentity';
    inputs = outputs;
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
    
    %% transformation 1
    nameLayer = 'transformation1';
    inputs = mapsOutputs1(layersNames1{end});
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
    outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));
    
	representationsTransform = cat(2, representationsTransform, outputs);
    
    fname = iLab_dagnn_setParamNameFilter(nameLayer);
    bname = iLab_dagnn_setParamNameBias(nameLayer); 
    in = outputsize(3);
    out = nTransformation;
	bias = zeros(out,1, 'single');
    filter = [zeros(in-out, out, 'single');  diag(ones(1,out, 'single'))]; 
    filter = reshape(filter, [1 1 in out]);
    
    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                {'size', [1 1 in out], 'stride', 1, 'pad', 0, ...
                'filter',           filter, ...
                'bias',             bias, ...
                'fname',            fname, ...
                'bname',            bname, ...
                'learningRate',     opts.conv.learningRate, ...
                'weightDecay',      opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale',            opts.conv.scale});
            
    lindex        =  net.getLayerIndex(nameLayer);       
    pindex        =  net.layers(lindex).paramIndexes;
    noUpdateLists =  cat(1, noUpdateLists, pindex(:));            
            
    %% transformation prediction1 
	nameLayer = 'predictionTransform';    
    inputs = outputs;
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
    
    
    %% identity2
    nameLayer = 'identity2';
    inputs    = mapsOutputs2(layersNames2{end});
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
    outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));

	representationsIdentity = cat(2, representationsIdentity, outputs);
    
    fname = iLab_dagnn_setParamNameFilter(nameLayer);
    bname = iLab_dagnn_setParamNameBias(nameLayer); 
    in = outputsize(3);
    out = nIdentity;
	bias = zeros(out,1, 'single');
    filter = [diag(ones(1,out, 'single'));  zeros(in-out, out, 'single')]; 
    filter = reshape(filter, [1 1 in out]);

    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                {'size', [1 1 in out], 'stride', 1, 'pad', 0, ...
                'filter',           filter, ...
                'bias',             bias, ...
                'fname',            fname, ...
                'bname',            bname, ...
                'learningRate',     opts.conv.learningRate, ...
                'weightDecay',      opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale',            opts.conv.scale});  
            
    lindex        =  net.getLayerIndex(nameLayer);       
    pindex        =  net.layers(lindex).paramIndexes;
    noUpdateLists =  cat(1, noUpdateLists, pindex(:));            
    
    %% transformation2
    nameLayer = 'transformation2';
    inputs = mapsOutputs2(layersNames2{end});
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
    outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));

	representationsTransform = cat(2, representationsTransform, outputs);
    
    fname = iLab_dagnn_setParamNameFilter(nameLayer);
    bname = iLab_dagnn_setParamNameBias(nameLayer); 
    in = outputsize(3);
    out = nTransformation;
	bias = zeros(out,1, 'single');
    filter = [zeros(in-out, out, 'single');  diag(ones(1,out, 'single'))]; 
    filter = reshape(filter, [1 1 in out]);
    
    net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                {'size', [1 1 in out], 'stride', 1, 'pad', 0, ...
                'filter',           filter, ...
                'bias',             bias, ...
                'fname',            fname, ...
                'bname',            bname, ...
                'learningRate',     opts.conv.learningRate, ...
                'weightDecay',      opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale',            opts.conv.scale});
    
	lindex        =  net.getLayerIndex(nameLayer);       
    pindex        =  net.layers(lindex).paramIndexes;
    noUpdateLists =  cat(1, noUpdateLists, pindex(:));  
    %% transformation prediction2         
	nameLayer = 'predictionTransform2';
    inputs = outputs;
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
    %% add 2 l2 loss layers, using identity representations from both images
    %% left  image identity loss
	nameLayer    = 'lossL2l';
    varObjective = 'objectiveL2l';    
    nameLayer    = iLab_dagnn_getNewLayerName(net, nameLayer);
	varObjective = iLab_dagnn_getNewVarName(net, varObjective);
    
    inputs = {representationsIdentity{1}, representationsIdentity{2}};
    outputs   =  varObjective;    
    net.addLayer( nameLayer, ...
                  dagnn.Loss('loss', 'l2'), ...
                  inputs, ...
                  outputs);
              
    outputsNames = cat(2, outputsNames, varObjective);    
    %% right image identity loss
    nameLayer    = 'lossL2r';
    varObjective = 'objectiveL2r';    
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
	varObjective = iLab_dagnn_getNewVarName(net, varObjective);
    
    inputs = {representationsIdentity{2}, representationsIdentity{1}};
    outputs   =  varObjective;    
    net.addLayer( nameLayer, ...
                  dagnn.Loss('loss', 'l2'), ...
                  inputs, ...
                  outputs);
    outputsNames = cat(2, outputsNames, varObjective);
    
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
    
    % get the updatelists
    net = iLab_dagnn_getParamUpdationLists(net);    
    paramIndex = 1:numel(net.params);
    noUpdateLists1 = setdiff(paramIndex, net.updatelists);
    noUpdateLists2 = noUpdateLists;
    updatelists = setdiff(paramIndex, [noUpdateLists1(:) ; noUpdateLists2(:)]);
    net.updatelists = updatelists;
    
    net.inputsNames = inputsNames;
    net.predictionsNames = predictionsNames;
    net.outputsNames = outputsNames;
    
    
    %% update synchronization lists: (layerlist, size, rate)
    % compute the size of the dropout layer
    % its size is equal to the size the input variable
    if ~isempty(net.synLayers)
        varSizes =   net.getVarSizes({mapsInputs1(layersNames1{1}), [inputImgSize 1], ...
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
    
end