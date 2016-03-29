
function locID =  vp_generate_Location_ID(fileNames)
    
    nimgs = numel(fileNames);
    lons = cell(nimgs, 1);
    lats = cell(nimgs, 1);
    IDs = cell(nimgs,1);
    
    parfor i=1:nimgs
        [lons{i}, lats{i}]= vp_get_LonLat(fileNames{i});        
        IDs{i} = [lons{i} lats{i}];
    end
    
    [~, ~, locID] = unique(IDs);
    
end