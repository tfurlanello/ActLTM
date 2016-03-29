function [net, net_linear] = ...
        iLab_arc_dagnn_multiLevelInjection_fc2(nclasses_obj, nclasses_env, ...
                                                args, baseModelType, labelgraph_env)

% architecture:
% inject labels into the top layer
    if ~exist('baseModelType', 'var') || isempty(baseModelType)
        baseModelType = 'alexnet';
    end
    
    if ~exist('labelgraph_env', 'var') || isempty(labelgraph_env)
        labelgraph_env = struct('isstructured', false, 'labelgraph', []);
    end                                            
                                            
    [net, net_linear] = ...
        iLab_arc_dagnn_2labelLayers_objenv(nclasses_obj, nclasses_env, args, ...
                                            baseModelType, labelgraph_env);
                                            
end