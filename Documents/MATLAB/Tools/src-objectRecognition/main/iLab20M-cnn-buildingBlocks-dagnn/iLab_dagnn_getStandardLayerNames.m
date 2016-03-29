function layerNames =  iLab_dagnn_getStandardLayerNames
    
    layerNames = struct('conv',     'conv', ...
                        'fc',       'fc', ...
                        'dropout',  'dropout', ...
                        'norm',     'norm', ...
                        'bnorm',    'bnorm', ...
                        'relu',     'relu', ...
                        'pool',     'pool', ...
                        'input', 'input');                    
    
end