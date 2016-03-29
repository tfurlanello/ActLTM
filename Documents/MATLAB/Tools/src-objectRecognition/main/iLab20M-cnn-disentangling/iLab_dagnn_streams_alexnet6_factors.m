function varargout = iLab_dagnn_streams_alexnet6_factors(nstreams, bshare_s, ...
                                          	 nfactors, nfractions, bshare_f, args)  

%% functionality
% build several parallel streams, using classic alexnet
% in default, all parallel streams share the same filters
% but, we could build individual streams using different parameter settings

%% notes:
% 1.
% this architecture has only 5 convolutional layers and 1 fully connected
% layer (this where 6 comes from ), 
% while the function 'iLab_dagnn_streams_alexnet.m' builds an architecture,
% which has 5 convolutional layers and 2 fully connected layers.

% 2.
% there are 'nfactors' stacked on the top of 'fc6'. These factors are
% latent factors, which explain the generation process of the observations.

%% inputs
%       nstream     - # of parallel streams
%       bshare_s    - wheather to share parameters or nor (true in
%                      default)
%       nfactors    - # of latent factors
%       nfractions  - # of nodes (in percentage) of each latent factor
%       bshare_f    - wheather to share the parameters or not
%       args        - it contains default settings for parameter
%                     intializations

%% outputs:
%       net             - a dagnn architecture 
%       LayerNames      - a linear chain base architecture, with 6 layers
%                         (either alexnet, or vgg-m)
%       mapsInputs      - a map data-structure, maps layer to input
%       mapsOutputs     - a map data structure, maps layer to output    
%       lf_LayerNames   - latent factor layers 
%       lf_mapsInputs   - map data-structure, maps latent layer to input
%       lf_mapsOutputs  - map data-structure, maps latent layer to output

%% note:
% the output of the architecture is the prediction
% don't add loss layer as the last layer; we will have much more
% flexibility to use different manually-crafted loss functions 

    narginchk(0,6);
    if ~exist('nstreams', 'var') || isempty(nstreams)
        nstreams = 2;
    end
    
    if ~exist('bshare_s', 'var') || isempty(bshare_s)
        bshare_s = true;
    end
    
    if ~exist('nfactors', 'var') || isempty(nfactors)
        nfactors = 3;
    end
    
    if ~exist('bshare_f', 'var') || isempty(bshare_f)
        bshare_f = true;
    end
    
    if ~exist('nfractions', 'var') || isempty(nfractions) || ...
            ~isvector(nfractions) || (numel(nfractions) ~= nfactors)
        nfractions = ones(1, nfactors) * (1/nfactors);
    end
    
    if ~exist('args', 'var') || isempty(args)
        args = {};
    end

    opts.batchNormalization = false ;
    
    opts.conv.weightInitMethod = 'gaussian';
    opts.conv.scale         = 1.0;
    opts.conv.learningRate  = [1 2];
    opts.conv.weightDecay   = [1 0];
	opts.fc.size            = 1024;

    
    opts.bnorm.learningRate = [2 1];
    opts.bnorm.weightDecay  = [0 0];
    
    opts.norm.param         = [5 1 0.0001/5 0.75];
    opts.pooling.method     = 'max';
    opts.dropout.rate       = 0.5;
    
    opts = vl_argparse(opts, args) ;
    
    fcsize = opts.fc.size;

    
    %% first build linear-streams architecture 
    [net, LayerNames, mapsInputs, mapsOutputs] = ...
                    iLab_dagnn_streams_alexnet6(nstreams, bshare_s, opts);
                
    %% then stack latent factors onto the top of each linear branch
    % latent factor layer (fully connected): convolution, activation, dropout    
	namesStandard =  iLab_dagnn_getStandardLayerNames;
    lfName        =  iLab_getLatentFactorName;
    nUnits        =  round(fcsize * nfractions);
    nUnits(end)   =  fcsize - sum(nUnits(1:end-1));
    
    lf_LayerNames      =   cell(nstreams,1);
    lf_LayerInputs     =   cell(nstreams,1);
    lf_LayerOutputs    =   cell(nstreams,1);
    conv_fnames  = cell(nfactors,1);
    conv_bnames  = cell(nfactors,1);
    bnorm_wnames = cell(nfactors,1);
    bnorm_bnames = cell(nfactors,1);
    
    synLayers  = cell(nfactors*10,1); % make sure 10 is greater than the #
                                      % the dropout layers in each factor
    cntDropout = zeros(nfactors,1);
    
    for s=1:nstreams
        
        cntDropout = zeros(nfactors,1);
        for nf=1:nfactors
            
            nthFactor = nf;
            nameLayer       = sprintf('%s%s%d', namesStandard.('fc'),   lfName,    nthFactor);
            nameBnorm       = sprintf('%s%s%d', namesStandard.('bnorm'), lfName,   nthFactor);
            nameActivation  = sprintf('%s%s%d', namesStandard.('relu'),  lfName,   nthFactor);
            nameDropout     = sprintf('%s%s%d', namesStandard.('dropout'), lfName, nthFactor);

            inputs = mapsOutputs{s}(LayerNames{s}{end});
            nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
            outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));

            lf_LayerNames{s}  = cat(2,lf_LayerNames{s}, {nameLayer}); 
            lf_LayerInputs{s} = cat(2,lf_LayerInputs{s}, {inputs});

            fname = iLab_dagnn_setParamNameFilter(nameLayer);
            bname = iLab_dagnn_setParamNameBias(nameLayer); 
            if s==1
                conv_fnames{nf} = fname; conv_bnames{nf} = bname;
            end
            if bshare_f
                fname = conv_fnames{nf};
                bname = conv_bnames{nf};       
            end
            filter = []; bias = [];

            net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                        {'size', [1 1 fcsize nUnits(nf)], 'stride', 1, 'pad', 0, ...
                        'filter',           filter, ...
                        'bias',             bias, ...
                        'fname',            fname, ...
                        'bname',            bname, ...
                        'learningRate',     opts.conv.learningRate, ...
                        'weightDecay',      opts.conv.weightDecay, ...
                        'weightInitMethod', opts.conv.weightInitMethod, ...
                        'scale',            opts.conv.scale});                 

            if opts.batchNormalization
                inputs = outputs;
                nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
                outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));

                wname = iLab_dagnn_setParamNameBNmean(nameBnorm);
                bname = iLab_dagnn_setParamNameBias(nameBnorm);
                if s==1
                    bnorm_wnames{nf} = wname; bnorm_bnames{nf} = bname;
                end            
                if bshare_f
                    wname = bnorm_wnames{nf};
                    bname = bnorm_bnames{nf};
                end
                weight = []; bias = [];

                net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                        {'nchannels',    nUnits(nf), ...
                                        'weight',       weight, ...
                                        'bias',         bias, ...
                                        'wname',        wname, ...
                                        'bname',        bname, ...                                    
                                        'learningRate', opts.bnorm.learningRate, ...
                                        'weightDecay',  opts.bnorm.weightDecay});                                                 
            end    

            inputs = outputs;
            nameActivation = iLab_dagnn_getNewLayerName(net, nameActivation);
            outputs        = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameActivation));
            net = iLab_dagnn_addlayer_activation(net, nameActivation, inputs, outputs, {'type', 'relu'});

            inputs = outputs;
            nameDropout = iLab_dagnn_getNewLayerName(net, nameDropout);
            outputs     = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameDropout));
            net = iLab_dagnn_addlayer_dropout(net, nameDropout, inputs, outputs, {'rate', opts.dropout.rate});

            %% synchronize dropout layers Feb. 18, 2016
            cntDropout(nf) = cntDropout(nf) + 1;
            idx = sum(cntDropout(1:nf));            
            l = net.getLayerIndex(nameDropout);
            synLayers{idx} = cat(2, synLayers{idx}, l);
            
            lf_LayerOutputs{s}  = cat(2,lf_LayerOutputs{s}, {outputs}); 
            
        end

    end

    lf_mapsInputs = cell(nstreams,1);
    lf_mapsOutputs = cell(nstreams,1);
    
    for s=1:nstreams
        lf_mapsInputs{s}   =   containers.Map(lf_LayerNames{s}, lf_LayerInputs{s});
        lf_mapsOutputs{s}  =   containers.Map(lf_LayerNames{s}, lf_LayerOutputs{s});
    end
    
    net.synLayers = cat(1, net.synLayers, synLayers(1:sum(cntDropout)));
    
    varargout = {net, LayerNames, mapsInputs, mapsOutputs, ...
                    lf_LayerNames, lf_mapsInputs, lf_mapsOutputs};

    
end