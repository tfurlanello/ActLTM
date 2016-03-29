function par_pool = iLab_layerpool(param)

    if nargin ==0 || ~exist('param', 'var') 
        par_pool = struct('type',   'pool', ...
                           'name',  '', ...
                           'method', 'max', ...
                           'pool',   [2 2], ...
                           'stride', 2, ...
                           'pad',    0) ;                       
        return;
    end
    
    if ~isstruct(param)
        error('only support struct input\n');
    end
    
    fields = {'type', 'name', 'method', 'pool', 'stride', 'pad'};
    values = {'pool', '', 'max', [2 2], 2, 0};
    
    par_pool = param;
    for f=1:numel(fields)
        if ~isfield(par_pool, fields{f})
            par_pool.(fields{f}) =  values{f};
        end
    end

end