function val_param = iLab_dagnn_validateNormParam(param)
    if nargin == 0 || ~exist('param', 'var') || isempty(param)
        val_param = struct( ...
                        'param', [5 1 0.0001/5 0.75]);
        return;
    end
    
    param = iLab_arg2struct(param);
    val_param = param;
    if ~isfield(val_param, 'param')
        val_param.param = [5 1 0.0001/5 0.75];
    end


end