
function instanceDir =  iLab_getInstanceRoot(args)  
 
%   classIdx, instanceIdx
%     rootdir = getRootiLab;
% 	classNames = getClassNameiLab;
%     if classIdx < 1 || classIdx > length(classNames)
%         fprintf(1, 'classIdx should lie within (%d %d)\n', 1, length(classNames));
%         instanceDir = [];
%         return;
%     end
%     instanceNames = getInstancesiLab(classIdx);
%     if instanceIdx < 1 || instanceIdx > length(instanceNames)
%         fprintf(1, 'class: %s\n instanceIdx should lie within (%d %d)\n', ...
%                           classNames{classIdx},  1, length(instanceNames));
%         instanceDir = [];
%         return;
%     end
%     instanceDir = fullfile(rootdir, classNames{classIdx}, instanceNames{instanceIdx});

    imgfilePara = iLab_validateImgFilePara;
    imgfilePara = vl_argparse(imgfilePara, args);
    
    iLabRoot = iLab_getRoot;   
    classFolderName = imgfilePara.class;
    
%     classes = getClassesiLab;
%     idx = strfind(classNames, classFolderName);
    instanceFolderName = iLab_genInstanceFolderName({'class', imgfilePara.class, ...
                                                    'instance', imgfilePara.instance});
    instanceDir = fullfile(iLabRoot, classFolderName, instanceFolderName);
    
end