function val_model = validateBoWmodel(model)
    
    if nargin == 0 || isempty(model) || ~isstruct(model)
        val_model = struct('clusterCenters', []);
        return;
    end
    
    if ~isfield(model, 'clusterCenters')
        val_model.clusterCenters = [];
    else
        val_model.clusterCenters = model.clusterCenters;
    end

    
end