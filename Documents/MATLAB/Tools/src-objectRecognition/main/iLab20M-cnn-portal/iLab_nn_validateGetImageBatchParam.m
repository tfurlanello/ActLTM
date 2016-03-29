function val_param = iLab_nn_validateGetImageBatchParam(param)

    if nargin == 0 || ~exist('param', 'var') || isempty(param)
        val_param = struct( ...
            'imageSize',        [227 227], ...
            'border',           [27 27], ...
            'keepAspect',       true, ...
            'numAugments',      1, ...
            'transformation',   'none', ...
            'averageImage',     [], ...
            'rgbVariance',      zeros(0,3, 'single'), ...
            'interpolation',    'bilinear', ...
            'numThreads',       1, ...
            'prefetch',         false);
        
        return;
    end

    val_param = iLab_arg2struct(param);
    
    fields={'imageSize'
            'border'
            'keepAspect'
            'numAugments'
            'transformation'
            'averageImage'
            'rgbVariance'
            'interpolation'
            'numThreads'
            'prefetch'};
            

    values={[227 227], [27 27], true, 1, 'none', [], zeros(0,3,'single'), 'bilinear', ...
            1, false};
    
    for f=1:numel(fields)
        if ~isfield(val_param, fields{f})
            val_param.(fields{f}) = values{f};
        end
    end


end