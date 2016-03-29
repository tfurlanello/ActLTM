function val_param = iLab_dagnn_validateLossParam(param)
    if nargin == 0 || ~exist('param', 'var') || isempty(param)
        val_param = struct(...
                    'type', 'softmaxlog', ...
                    'isstructured', false, ...
                    'labelgraph', []);
        return;
    end
    
    param = iLab_arg2struct(param);
    val_param = param;
    fields = {'loss', 'isstructured', 'labelgraph'};
    values = {'softmaxlog', false, []};
    
    for f=1:numel(fields)
        if ~isfield(val_param, fields{f})
            val_param.(fields{f}) = values{f};
        end
    end
    
    
end