
function imgHierarchy = iLab_genImgHierarchy
    imgsPara = iLab_getImgParameters;
    
    classes = imgsPara.classes;
    nClasses = numel(classes);
    
    imgHierarchy.nclasses = nClasses;
    imgHierarchy.classes = classes;
    
    % category level
    for c=1:nClasses
        bClass = imgsPara.objectsIdx == c;
        instances = unique(imgsPara.instancesIdx(bClass));
        nInstances = numel(instances);
        className = classes{c};
        
        % instance level
        for i=1:nInstances
            bInstance = imgsPara.instancesIdx == instances(i);
            instanceName = iLab_idx2nameInstance(instances(i));
            
            b = bClass & bInstance;
            backgrounds = unique(imgsPara.backgroundsIdx(b));            
            nBackgrounds = numel(backgrounds);
            
            imgHierarchy.(className).(instanceName).nbackgrounds = nBackgrounds;
            imgHierarchy.(className).(instanceName).backgrounds = backgrounds;
        end 
        imgHierarchy.(className).instances = instances;
    end
    
    save('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-data-info/imagefiles-hierarchy.mat', 'imgHierarchy');
    
end