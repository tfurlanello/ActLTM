function net = iLab_simplenn_vgg_m(varargin)
% --------------------------------------------------------------------
% vgg-m    
    
    opts.scale = 1 ;
    opts.initBias = 0.1 ;
    opts.weightDecay = 1 ;
    %opts.weightInitMethod = 'xavierimproved' ;
    opts.weightInitMethod = 'gaussian' ;
    opts.model = 'vgg-m' ;
    opts.batchNormalization = false ;
    opts.nclasses = 1000;
    opts = vl_argparse(opts, varargin) ;
    

    net.normalization.imageSize = [224, 224, 3] ;
    net.layers = {} ;
    net = iLab_simplenn_addblock(net, opts, '1', 7, 7, 3, 96, 2, 0) ;
    net = iLab_simplenn_addnorm(net, opts, '1') ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = iLab_simplenn_addblock(net, opts, '2', 5, 5, 96, 256, 2, 1) ;
    net = iLab_simplenn_addnorm(net, opts, '2') ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', [0 1 0 1]) ;

    net = iLab_simplenn_addblock(net, opts, '3', 3, 3, 256, 512, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '4', 3, 3, 512, 512, 1, 1) ;
    net = iLab_simplenn_addblock(net, opts, '5', 3, 3, 512, 512, 1, 1) ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = iLab_simplenn_addblock(net, opts, '6', 6, 6, 512, 1024, 1, 0) ;
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
