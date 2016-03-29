function [net, outputs] = iLab_dagnn_addlayer_fc(net, inputSize, outputSize, ...
                                                    layerNamePrefix, inputs, args) 	

	opts.batchNormalization = false ;
    
    opts.conv.weightInitMethod = 'gaussian';
    opts.conv.scale         = 1.0;
    opts.conv.learningRate  = [1 2];
    opts.conv.weightDecay   = [1 0];
    opts.fc.size             = 1024;
    
    opts.bnorm.learningRate = [2 1];
    opts.bnorm.weightDecay  = [0 0];
    
    opts.norm.param         = [5 1 0.0001/5 0.75];
    opts.pooling.method     = 'max';
    opts.dropout.rate       = 0.5;
    
    opts = vl_argparse(opts, args) ;


    nameLayer       =   [layerNamePrefix 'fc'];
    nameBnorm       =   [layerNamePrefix 'bnorm'];
    nameActivation  =   [layerNamePrefix 'relu'];
    nameDropout     =   [layerNamePrefix 'dropout'];
        
    nameLayer = iLab_dagnn_getNewLayerName(net, nameLayer);
    outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameLayer));
    
	net = iLab_dagnn_addlayer_conv(net, nameLayer, inputs, outputs, ...
                {'size', [inputSize(1:3) outputSize], 'stride', 1, 'pad', 0, ...
                'learningRate', opts.conv.learningRate, ...
                'weightDecay',  opts.conv.weightDecay, ...
                'weightInitMethod', opts.conv.weightInitMethod, ...
                'scale', opts.conv.scale});
            
    if opts.batchNormalization
      inputs = outputs;
      nameBnorm = iLab_dagnn_getNewLayerName(net, nameBnorm);
      outputs   = iLab_dagnn_getNewVarName(net, sprintf('%sout', nameBnorm));
      net = iLab_dagnn_addlayer_bnorm(net, nameBnorm, inputs, outputs, ...
                                {'nchannel', outputSize, 'learningRate', opts.bnorm.learningRate, ...
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
        
    
end