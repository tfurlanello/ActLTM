function mapLabels =  iLab_de_genLabelSpace_f7
   
    %% we take adjacent cameras under the same rotation as a pair

    rotations = iLab_getRotations;    
    nRotations = numel(rotations);
    
    labelKeys = {};
    labelValues = [];
    separator = iLab_de_getDash;
    
    %%  the same rotation, but adjacent cameras (skip 2)
    skip = 2;
    neighboringCameras = iLab_de_getNeighboringCameras(skip);
    nCameraPairs = size(neighboringCameras,1);
    
    for p=1:nCameraPairs        
        c1 = neighboringCameras(p,1);
        c2 = neighboringCameras(p,2);
        
        l_c1 = iLab_idx2nameCamera(c1);
        l_c2 = iLab_idx2nameCamera(c2);
        
        for r=1:nRotations
            ref_r = rotations(r);
            l_r = iLab_idx2nameRotation(ref_r);
            
            l = [l_c1 separator l_c2 separator l_r];
            
            labelKeys = cat(2, labelKeys, l);
            labelValues = cat(2, labelValues, p); 
            
        end
    end
    
    mapLabels = containers.Map(labelKeys, labelValues);
    
end