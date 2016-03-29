function net = iLab_simplenn_addnorm(net, opts, id)
% --------------------------------------------------------------------
opts = iLab_validateCNNnetParam(opts);
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;
end