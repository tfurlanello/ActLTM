function val_param = iLab_simplenn_validatePoolingParam(param)
    if nargin == 0 || ~exist('param', 'var') || isempty(param) 
        val_param = struct('method', 'max', ...
                            'pool', [3 3], ...
                            'stride', 2, ...
                            'pad', 0);
        return;
    end
    
    param = iLab_arg2struct(param);
    fields = {'method', 'pool', 'stride', 'pad'};
    values = {'max', [3 3], 2, 0};
    
    val_param = param;
    for f=1:numel(fields)
        if ~isfield(val_param, fields{f})
            val_param.(fields{f}) = values{f};
        end
    end
    
    
end