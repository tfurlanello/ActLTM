% visualize data, to see whether the spatial relationship between the
% camera and the initial oritation of objects are the same under different
% backgrounds


objects = {'f1car',  'mil', 'tank', 'van'}; % now test only under 
                                                  % 5 object classes
focus               = 1;    % now only use images with focus 1
light               = 0;    % only use 1 light
rotation = 0; 

instancesIdx = 1:40;
                          
imgFileNames = getImgFileNamesiLab;

%% (1) filter images
iLabInfo = getImgParametersiLab;
nImages     = length(iLabInfo.objectsIdx);

% filter focus
fFocus = iLabInfo.focusIdx == focus;
% filter light
fLight = iLabInfo.lightsIdx == light;
% filter rotation
fRotation = iLabInfo.rotationsIdx == rotation;

% filter classes
cnt = 0;
flags = {};
for i=1:length(objects)
    i_object = objects{i};    
    classIdx = getClassIdxiLab(i_object);    
    fClass = classIdx == iLabInfo.objectsIdx;
    
    for j=1:numel(instancesIdx)
        instanceIdx = instancesIdx(j);
        fInstance = iLabInfo.instancesIdx == instanceIdx;
        if sum(fClass & fInstance) == 0
            continue;
        end
        
        b = fClass & fInstance & fFocus & fLight & fRotation;
        cnt = cnt + 1;
        
        selected.imgNames   = imgFileNames(b);
        selected.imgDirsIdx  = iLabInfo.imgDirsIdx(b);
        selected.imgDirs    = iLabInfo.imgDirs;

        
        flags{cnt} = selected;
    end
     
end

save('data-relativeOrientation-camins.mat', 'flags', '-v7.3');






