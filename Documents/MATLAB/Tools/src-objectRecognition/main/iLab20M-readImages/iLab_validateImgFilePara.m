
function val_param = iLab_validateImgFilePara(param)

    if nargin == 0 || ~exist('param', 'var') || isempty(param)
        val_param = struct('class',       'car', ...
                 'instance',    160, ...
                 'background',  1, ...
                 'camera',      0, ... 
                 'rotation',    0, ...
                 'light',     0, ...
                 'focus',       1);
             return;
    end
    
    val_param = param;
    fields = {'class', 'instance', 'background', ...
                    'camera', 'rotation', 'light', 'focus'};
    fieldsvals = {'van', 1,1,1,1,0,1};
    
    for i=1:numel(fields)
        if ~isfield(val_param, fields{i})
            val_param.(fields{i}) = fieldsvals{i};            
        end
    end
     
end