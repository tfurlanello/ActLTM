function val_param = iLab_dagnn_validateDropoutParam(param)
    if nargin == 0 || ~exist('param', 'var') || isempty(param)
        val_param = struct('rate', 0.5, ...
                            'frozen', false);
        return;
    end

    param = iLab_arg2struct(param);
    val_param = param;
    fields = {'rate', 'frozen'};
    values = {0.5, false};
    
    for f=1:numel(fields)
        if ~isfield(val_param, fields{f})
            val_param.(fields{f}) = values{f};
        end
    end 
end