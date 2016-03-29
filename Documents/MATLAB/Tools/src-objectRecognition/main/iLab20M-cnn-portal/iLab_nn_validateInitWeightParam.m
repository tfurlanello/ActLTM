function val_param = iLab_nn_validateInitWeightParam(param)
   
    if nargin == 0 || ~exist('param', 'var')
        val_param = struct( ...
                    'scale',            1, ...             
                    'weightInitMethod', 'gaussian');
                
        return;
    end
    
    param = iLab_arg2struct(param);
    val_param = param;
    fields = {'scale','weightInitMethod'};
    values = {1,'gaussian'};
            
    for f=1:numel(fields)
        if ~isfield(val_param, fields{f})
            val_param.(fields{f}) = values{f};
        end
    end    

    
    
end