function mapLabels =  iLab_de_genLabelSpace_f11
    
    %% we take two adjacent rotations under the same camera as a pair

    cameras = iLab_getCameras;
    rotations = iLab_getRotations;
    
    nCameras = numel(cameras);
    nRotations = numel(rotations);
    
    labelKeys = {};
    labelValues = [];
    separator = iLab_de_getDash;
  
    %% the same camera, with adjacent rotations  
    for c=1:nCameras       
        c_ref = cameras(c);        
        l_c = iLab_idx2nameCamera(c_ref);        
        for r=1:nRotations-1
            r1 = rotations(r);
            r2 = rotations(r+1);
            
            l_r1 = iLab_idx2nameRotation(r1);
            l_r2 = iLab_idx2nameRotation(r2);
            
            l = [l_c separator l_r1 separator l_r2];
            labelKeys = cat(2, labelKeys, l);
            labelValues = cat(2, labelValues, c);
        end        
    end
    
    mapLabels = containers.Map(labelKeys, labelValues);
    
end