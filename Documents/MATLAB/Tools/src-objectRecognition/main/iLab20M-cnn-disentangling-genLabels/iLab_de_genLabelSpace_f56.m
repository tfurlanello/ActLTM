function mapLabels =  iLab_de_genLabelSpace_f56
    
    cameras     = iLab_getCameras;
    rotations   = iLab_getRotations;
    
    nCameras    = numel(cameras);
    nRotations  = numel(rotations);
    
    labelKeys = {};
    labelValues = [];
    separator = iLab_de_getDash;
  
    %% case 1: the same camera, with adjacent rotations  
    for c=1:nCameras       
        c_ref = cameras(c);        
        l_c = iLab_idx2nameCamera(c_ref);        
        cntLabels = 2*(c-1)+1;
        for r=1:nRotations-1
            r1 = rotations(r);
            r2 = rotations(r+1);
            
            l_r1 = iLab_idx2nameRotation(r1);
            l_r2 = iLab_idx2nameRotation(r2);
            
            l = [l_c separator l_r1 separator l_r2];
            labelKeys = cat(2, labelKeys, l);
            labelValues = cat(2, labelValues, cntLabels);
        end        
        
        cntLabels = 2*c;
        for r=1:nRotations-2
            r1 = rotations(r);
            r2 = rotations(r+2);
            
            l_r1 = iLab_idx2nameRotation(r1);
            l_r2 = iLab_idx2nameRotation(r2);
            
            l = [l_c separator l_r1 separator l_r2];
            labelKeys = cat(2, labelKeys, l);
            labelValues = cat(2, labelValues, cntLabels);
        end
    end
    cntLabels = 2*nCameras;
    
    %% case 2: the same rotation, but adjacent cameras    
    mapNeiCams_s = iLab_de_getNeighboringCameras_sameRot;
    cams_ref = mapNeiCams_s.keys;
    cams_tar = mapNeiCams_s.values;
    nCameraPairs = numel(cams_ref);
    
    for p=1:nCameraPairs        
        
        c_ref = cams_ref{p};
        c_tars = cams_tar{p};
        ntars = numel(c_tars);        
        
        for c=1:ntars
            
            c_tar = c_tars(c);
            l_c1 = iLab_idx2nameCamera(c_ref);
            l_c2 = iLab_idx2nameCamera(c_tar);
            
            cntLabels = cntLabels + 1;
            
            for r=1:nRotations
                ref_r = rotations(r);
                l_r = iLab_idx2nameRotation(ref_r);
                l = [l_c1 separator l_c2 separator l_r];
                labelKeys = cat(2, labelKeys, l);
                labelValues = cat(2, labelValues, cntLabels); 

            end        
        end
    end
    
    
    %% case 3: the neighboring cameras, but adjacent rotations    
    mapNeiCams_d = iLab_de_getNeighboringCameras_diffRot;
    cams_mid     = mapNeiCams_d.keys;
    cams_side    = mapNeiCams_d.values;    
    nPairs       = numel(cams_mid);
    
    for p=1:nPairs   
        
        cam_mid = cams_mid{p};
        cam_side = cams_side{p};        
        nSides = numel(cam_side);        
        l_c_ref = iLab_idx2nameCamera(cam_mid);
        
        for c=1:nSides
            c_cam_side = cam_side(c);
            l_c_tar = iLab_idx2nameCamera(c_cam_side);
            cntLabels = cntLabels + 1;
            for r=1:nRotations-1
               ref_r = rotations(r);
               tar_r = rotations(r+1);
                
               l_r_ref = iLab_idx2nameRotation(ref_r);
               l_r_tar = iLab_idx2nameRotation(tar_r);
               
               
                l = [l_c_ref separator l_c_tar separator ...
                            l_r_ref separator l_r_tar];
                labelKeys = cat(2, labelKeys, l);
                labelValues = cat(2, labelValues, cntLabels); 
                
                
                l = [l_c_tar separator l_c_ref separator ...
                            l_r_tar separator l_r_ref];
                labelKeys = cat(2, labelKeys, l);
                labelValues = cat(2, labelValues, cntLabels);                 
                
            end
        end
    end
    
    mapLabels = containers.Map(labelKeys, labelValues);
    
end