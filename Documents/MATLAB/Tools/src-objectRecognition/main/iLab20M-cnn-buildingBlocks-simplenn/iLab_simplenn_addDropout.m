function net = iLab_simplenn_addDropout(net, opts, id)
% --------------------------------------------------------------------
opts = iLab_validateCNNnetParam(opts);
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end