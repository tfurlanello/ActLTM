function idx = iLab_getClassIdx(className)
    classNames = iLab_getClasses;
    
    fExist = false;
    for i=1:length(classNames)
        if strcmpi(className, classNames{i})
            idx = i;
            fExist = true;
            break;
        end
    end
    
    if fExist == false
        error('%s doesn''t exist\n', className);
    end
    
    
end