function val_param = iLab_simplenn_validateBnormParam(param)
    if nargin == 0 || ~exist('param', 'var') || isempty(param)
        val_param = struct(...
                           'nchannels',     96, ...
                           'weight',        ones(96, 1, 'single'), ...
                           'bias',          zeros(96,1, 'single'), ...
                           'learningRate',  [2 1], ...
                           'weightDecay',   [0 0]);
        return;
                            
    end
    
    param = iLab_arg2struct(param);
    val_param = param;
    
    fields = {'nchannels', 'weight', 'bias', 'learningRate', 'weightDecay'};
    values = {96, ones(96,1, 'single'), zeros(96,1,'single'), [2 1], [0 0]};
    
    for f=1:numel(fields)
        if ~isfield(val_param, fields{f})
            switch fields{f}
                case 'weight'
                    val_param.weight = ones(val_param.nchannels, 1, 'single');
                case 'bias'
                    val_param.bias   = zeros(val_param.nchannels, 1, 'single');
                otherwise
                    val_param.(fields{f}) = values{f};
            end
        end
    end
        
    
    
end