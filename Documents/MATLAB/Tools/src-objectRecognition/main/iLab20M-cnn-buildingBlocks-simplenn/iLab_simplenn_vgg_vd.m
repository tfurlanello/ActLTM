function net = iLab_simplenn_vgg_vd(varargin)
% --------------------------------------------------------------------
% vgg-vd    
     
    opts.scale = 1 ;
    opts.initBias = 0.1 ;
    opts.weightDecay = 1 ;
    %opts.weightInitMethod = 'xavierimproved' ;
    opts.weightInitMethod = 'gaussian' ;
    opts.model = 'vgg-vd-16' ;
    opts.batchNormalization = false ;     
    opts = vl_argparse(opts, varargin) ;

	net.normalization.imageSize = [224, 224, 3] ;

    net.layers = {} ;
    net = iLab_simplenn_addblock(net, opts, '1_1', 3, 3, 3, 64, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '1_2', 3, 3, 64, 64, 1, 1) ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                               'method', 'max', ...
                               'pool', [2 2], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = iLab_simplenn_addblock(net, opts, '2_1', 3, 3, 64, 128, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '2_2', 3, 3, 128, 128, 1, 1) ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                               'method', 'max', ...
                               'pool', [2 2], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = iLab_simplenn_addblock(net, opts, '3_1', 3, 3, 128, 256, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '3_2', 3, 3, 256, 256, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '3_3', 3, 3, 256, 256, 1, 1) ;
    if strcmp(opts.model, 'vgg-vd-19')
      net = iLab_simplenn_addblock(net, opts, '3_4', 3, 3, 256, 256, 1, 1) ;
    end
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                               'method', 'max', ...
                               'pool', [2 2], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = iLab_simplenn_addblock(net, opts, '4_1', 3, 3, 256, 512, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '4_2', 3, 3, 512, 512, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '4_3', 3, 3, 512, 512, 1, 1) ;
    if strcmp(opts.model, 'vgg-vd-19')
      net = iLab_simplenn_addblock(net, opts, '4_4', 3, 3, 512, 512, 1, 1) ;
    end
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool4', ...
                               'method', 'max', ...
                               'pool', [2 2], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = iLab_simplenn_addblock(net, opts, '5_1', 3, 3, 512, 512, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '5_2', 3, 3, 512, 512, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '5_3', 3, 3, 512, 512, 1, 1) ;
    if strcmp(opts.model, 'vgg-vd-19')
      net = iLab_simplenn_addblock(net, opts, '5_4', 3, 3, 512, 512, 1, 1) ;
    end
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                               'method', 'max', ...
                               'pool', [2 2], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = iLab_simplenn_addblock(net, opts, '6', 7, 7, 512, 1024, 1, 0) ;
    net = iLab_simplenn_addDropout(net, opts, '6') ;

    net = iLab_simplenn_addblock(net, opts, '7', 1, 1, 1024, 1024, 1, 0) ;
    net = iLab_simplenn_addDropout(net, opts, '7') ;

    net = iLab_simplenn_addblock(net, opts, '8_1', 1, 1, 1024, opts.nclasses, 1, 0) ;
    net.layers(end) = [] ;
    if opts.batchNormalization, net.layers(end) = [] ; end

    % final touches
    switch lower(opts.weightInitMethod)
      case {'xavier', 'xavierimproved'}
        net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
    end
    net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

    net.normalization.border = 256 - net.normalization.imageSize(1:2) ;
    net.normalization.interpolation = 'bicubic' ;
    net.normalization.averageImage = [] ;
    net.normalization.keepAspect = true ;    
    
    
end