function par_conv =  iLab_layerconv(param)

    if nargin == 0 || isempty(param)
        par_conv = struct('type', 'conv', ...
                      'name', '', ...
                      'weights', {{randn(2, 2, 5, 10, 'single'), zeros(10, 1, 'single')}}, ...
                      'stride', 2, ...
                      'pad', 0, ...
                      'learningRate', [1 2], ...
                      'weightDecay', [1 0]) ;
        return;
    end

    fields = {'type', 'name', 'weights', 'stride', 'pad', 'learningRate', 'weightDecay'};
    values = {'conv', '', {{randn(2, 2, 5, 10, 'single'), zeros(10, 1, 'single')}}, 2, 0, [1 2], [1 0]};
    
    par_conv = param;
    for f=1:numel(fields)
        if ~isfield(par_conv, fields{f})
            par_conv.(fields{f}) = values{f};
        end
    end
    
    
end