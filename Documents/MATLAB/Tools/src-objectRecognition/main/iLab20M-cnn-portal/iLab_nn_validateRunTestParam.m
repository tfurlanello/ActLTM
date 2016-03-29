function val_param = iLab_nn_validateRunTestParam(param)
    
    
    if nargin == 0 || ~exist('param', 'var') || isempty(param)
        val_param = struct( ...
                        'batchSize',      128, ...
                        'numSubBatches',  1, ...
                        'gpus',           1, ...
                        'conserveMemory', false, ...
                        'sync',           false, ...
                        'prefetch',       false, ...
                        'cudnn',          true);
        return;                    
    end
    
    param = iLab_arg2struct(param);
    val_param = param;
    
    fields = {'batchSize', 'numSubBatches', 'gpus', 'conserveMemory', ...
              'sync', 'prefetch', 'cudnn'};
    values = {128, 1, 1, false, ...
              false, false, true};
          
    for f=1:numel(fields)
        if ~isfield(val_param, fields{f})
            val_param.(fields{f}) = values{f};
        end
        
    end

end