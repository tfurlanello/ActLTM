function net = iLab_cnnArchitecture_catcam(varargin)
    % obsolete, don't use again
    % initialize CNN architecture
    % label layer consists of two kinds of labels: category & camera
 
    opts.scale = 1 ;
    opts.initBias = 0.1 ;
    opts.weightDecay = 1 ;
    %opts.weightInitMethod = 'xavierimproved' ;
    opts.weightInitMethod = 'gaussian' ;
    opts.model = 'alexnet' ;
    opts.batchNormalization = false ;
    opts.nclasses = 16;
    opts = vl_argparse(opts, varargin) ;

    % Define layers
    switch opts.model
      case 'alexnet'
        net = iLab_simplenn_alexnet(opts) ;
      case 'vgg-f'
        net = iLab_simplenn_vgg_f(opts) ;
      case 'vgg-m'
        net = iLab_simplenn_vgg_m(opts) ;
      case 'vgg-s'
        net = iLab_simplenn_vgg_s(opts) ;
      case {'vgg-vd-16', 'vgg-vd-19'}              
        net = iLab_simplenn_vgg_vd(opts) ;
      otherwise
        error('Unknown model ''%s''', opts.model) ;
    end
    
	net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
                 {'prediction','label'}, 'top1error') ;

    
    %% now we add another loss layer, which has the same inputs as the 
    % 'fc8_1' layer, and with camera parameters as the outputs
% 	cam_layer =  iLab_simplenn_addblock(cam_layer, opts, '8_2', 1, 1, 4096, 88, 1, 0) ;
    cam_layer.layers = {};
    id = '8_2';
    h = 1; w = 1; in = 1024; out = 88;
    cam_layer.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', 'fc', id), ...
                           'weights', {{iLab_nn_initWeight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0]) ;

	block = dagnn.Conv() ;
    block.size = size(cam_layer.layers{1}.weights{1}) ;
    block.pad = cam_layer.layers{1}.pad ;
    block.stride = cam_layer.layers{1}.stride ;    
                       
	name = cam_layer.layers{1}.name ;   
    params = struct(...
        'name', {}, ...
        'value', {}, ...
        'learningRate', [], ...
        'weightDecay', []) ;
    params(1).name  = sprintf('%sf',name) ;
    params(1).value = cam_layer.layers{1}.weights{1} ;
    params(1).weightDecay  = cam_layer.layers{1}.weightDecay(1) ;
    params(1).learningRate = cam_layer.layers{1}.learningRate(1) ;
    params(2).name  = sprintf('%sb',name) ;
    params(2).value = cam_layer.layers{1}.weights{2} ;
	params(2).learningRate = cam_layer.layers{1}.learningRate(2) ;
    params(2).weightDecay  = cam_layer.layers{1}.weightDecay(2) ;
    
    idxLayer    = getLayerIndex(net, 'fc8_1');
    inputs      =  net.layers(idxLayer).inputs; 
    outputs     = {'prediction_camera'};
    
    net.addLayer(...
            name, ...
            block, ...
            inputs, ...
            outputs, ...
            {params.name}) ;
        
    findex = net.getParamIndex(params(1).name) ;
    bindex = net.getParamIndex(params(2).name) ;
    net.params(findex).value        = params(1).value ;
    net.params(findex).learningRate = params(1).learningRate ;
    net.params(findex).weightDecay  = params(1).weightDecay ;    
    net.params(bindex).weightDecay  = params(2).weightDecay ;
    net.params(bindex).value        = params(2).value ;
    net.params(bindex).learningRate = params(2).learningRate ;
    
    % add another loss layer on top of the camera prediction layer
    params = struct(...
                    'name', {}, ...
                    'value', {}, ...
                    'learningRate', [], ...
                    'weightDecay', []) ;
	params(1).name = sprintf('%s_isstructured', 'loss_camera');
    params(1).value = true;
    params(2).name = sprintf('%s_labelgraph', 'loss_camera');
    params(2).value = [];
    net.addLayer(...
                'loss_camera', ...
                dagnn.Loss('loss', 'softmaxlog') , ...
                {'prediction_camera', 'label_camera'}, ...
                'objective_camera', ...
                {params.name}) ;
    index = net.getParamIndex(params(1).name);
    net.params(index).value = params(1).value;
    
    
	params = struct(...
                    'name', {}, ...
                    'value', {}, ...
                    'learningRate', [], ...
                    'weightDecay', []) ;
	params(1).name = sprintf('%s_isstructured', 'error_camera');
    params(1).value = true;
    params(2).name = sprintf('%s_labelgraph', 'error_camera');
    params(2).value = [];
    net.addLayer('error_camera', ...
                dagnn.Loss('loss', 'classerror'), ...
                {'prediction_camera','label_camera'}, ...
                'top1error_camera', ...
                 {params.name}) ;  
	index = net.getParamIndex(params(1).name);
    net.params(index).value = params(1).value;

    
end
 
 
