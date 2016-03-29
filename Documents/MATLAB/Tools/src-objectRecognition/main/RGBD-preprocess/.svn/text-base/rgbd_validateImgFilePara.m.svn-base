
function val_param = rgbd_validateImgFilePara(param)

    if nargin == 0 || ~exist('param', 'var') || isempty(param)
        val_param = struct(...
                 'class',    'bowl', ...
                 'instance',  1, ...                 
                 'camera',    1, ... 
                 'frame',     1, ...
                 'center',    [], ...
                 'scale',     []); %
             return;
    end
    
    val_param = param;
    fields = {'class', 'instance', 'camera','frame', 'center', 'scale'};
    fieldsvals = {'bowl', 1,1,1, [], []};
    
    for i=1:numel(fields)
        if ~isfield(val_param, fields{i})
            val_param.(fields{i}) = fieldsvals{i};            
        end
    end
     
end