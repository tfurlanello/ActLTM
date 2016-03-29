function netParam = iLab_validateCNNnetParam(param)
    if nargin == 0 || ~exist('param', 'var')
        netParam = struct( ...
                    'scale',            1, ...
                    'initBias',         0.1, ...
                    'weightDecay',      1, ...                
                    'weightInitMethod', 'gaussian', ...
                    'model',            'vgg-vd-16', ...                                                        
                    'batchNormalization', false);
                
        return;
    end
    
    % model - vgg-f, vgg-m, vgg-s, vgg-vd-16, vgg-vd-19
    netParam = param;
    fields = {'scale', 'initBias', 'weightDecay', ...
                'weightInitMethod', 'model', 'batchNormalization'};
    values = {1, 0.1, 1, 'gaussian', 'vgg-vd-16', false};
            
    for f=1:numel(fields)
        if ~isfield(netParam, fields{f})
            netParam.(fields{f}) = values{f};
        end
    end

end