
function dirTree =  iLab_getDataDirTree

    dataroot        = iLab_getRoot;
    classNames      = iLab_getClasses;
    nClasses        = length(classNames);
    dirTree.root    = dataroot;
    dirTree.classes = classNames;
    
    instanceNames = cell(nClasses,1);
    for i=1:nClasses
        instanceNames{i} = iLab_getInstances(i);
    end    
     
    imageFilesiLab = {};
    % each instance, extract: background info
    classesInfo = struct('instances', {}, 'instancesInfo', {});
    for i=1:nClasses
        nInstances = length(instanceNames{i});
        c_instances = struct('instance', {}, 'instanceDir', {}, 'backgrounds', {}); 
        
        instancesInfo = struct('backgrounds', {}, 'backgroundsInfo', {});
        for j=1:nInstances
            instName = instanceNames{i}{j};
            instDir = fullfile(dataroot, classNames{i}, instName);
            backgrounds = getSubfolders(instDir);            
            c_instances(j).instance = instName;
            c_instances(j).instanceDir = instDir;
            c_instances(j).backgrounds = backgrounds;
            
            instancesInfo(j).backgrounds = backgrounds;
            backgroundsInfo = struct('imageNames', {}, 'imageFiles', {});
            for k=1:length(backgrounds)
                imageNames = getImgFiles(fullfile(instDir, backgrounds{k}), '.png');
                imageNames_n = cell(size(imageNames));
                imageFiles = cell(size(imageNames));
                for n=1:length(imageNames)
                    imageNames_n{n} = imageNames{n}(1:end-4);
                    imageFiles{n} = fullfile(instDir, backgrounds{k}, imageNames{n});
                end
                
                backgroundsInfo(k).imageNames = imageNames_n;
                backgroundsInfo(k).imageFiles = imageFiles;
                imageFilesiLab = cat(1, imageFilesiLab, imageFiles);
                
            end
            
            
            instancesInfo(j).backgroundsInfo = backgroundsInfo;
            
        end
%         classesInfo(i).instances = instanceNames{i};
%         classesInfo(i).instancesInfo = instancesInfo;
        
    end
    
%     dirTree.classesInfo = classesInfo;

    %% parse the imagefiles
    
    nImages = length(imageFilesiLab);
    imgDirs = cell(nImages,1);
    imgNames = cell(nImages,1);

    fprintf(1, 'parsing...\n');
    parfor i=1:nImages
        [imgDirs{i},imgNames{i},~] = fileparts(imageFilesiLab{i}); 
        imgNames{i} = [imgNames{i} '.png'];
    end

    fprintf(1, 'building index...\n');
    [u_imgDirs, idx, idx_u]   = unique(imgDirs);
 
    imgDirsIdx = idx;
    imgDirs = u_imgDirs;
    dirTree.imgDirsIdx  = imgDirsIdx;
    dirTree.imgDirs     = imgDirs;
    dirTree.imgNames    = imgNames;

    %% parse each file name    
    instancesIdx    = uint8(zeros(nImages,1));   
    backgroundsIdx  = uint16(zeros(nImages,1));
    lightsIdx       = uint8(zeros(nImages,1));
    focusIdx        = uint8(zeros(nImages,1));
    camerasIdx      = uint8(zeros(nImages,1));
    rotationsIdx    = uint8(zeros(nImages,1));
    objectsIdx      = uint8(zeros(nImages,1));

    fprintf(1, 'parsing file names...\n');
    parfor i=1:nImages
        fileName = imgNames{i};
        options =  iLab_parseImgName(fileName);

        instancesIdx(i)     =   uint8(options.instance);
        backgroundsIdx(i)   =   uint16(options.background);
        lightsIdx(i)        =   uint8(options.lighting);
        rotationsIdx(i)     =   uint8(options.rotation);
        camerasIdx(i)       =   uint8(options.camera);
        focusIdx(i)         =   uint8(options.focus);    
        objectsIdx(i)       =   uint8(iLab_getClassIdx(options.class));    

    end

    dirTree.instancesIdx    = instancesIdx;
    dirTree.backgroundsIdx  = backgroundsIdx;
    dirTree.lightsIdx       = lightsIdx;
    dirTree.rotationsIdx    = rotationsIdx;
    dirTree.camerasIdx      = camerasIdx;
    dirTree.focusIdx        = focusIdx;
    dirTree.objectsIdx      = objectsIdx;

    
end


