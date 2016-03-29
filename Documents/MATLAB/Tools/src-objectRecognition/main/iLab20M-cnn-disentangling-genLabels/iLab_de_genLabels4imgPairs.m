function varargout = iLab_de_genLabels4imgPairs(arg1, arg2)
% functionality: generate labels for input image pairs
% inputs:
%         arg1 - the 1st image (name)
%         arg2 - the 2nd image (name)


    arg1 =   vl_argparse(iLab_validateImgFilePara, arg1);
    arg2 =   vl_argparse(iLab_validateImgFilePara, arg2);
    
    assert( strcmp(arg1.class, arg2.class));
    assert( arg1.instance == arg2.instance);
    
    c1 = arg1.camera;
    r1 = arg1.rotation;
    
    c2 = arg2.camera;
    r2 = arg2.rotation;
    
    separator = iLab_de_getDash;
    
    if (c1 ~= c2 && r1 ~= r2) || ...
            (c1 == c2 && r1 == r2)
        class = {};
        transform = {};
        varargout = {class, transform};
        return;
    end
    
    class = arg1.class;
    if c1 == c2 && r1 ~= r2        
        l_c = iLab_idx2nameCamera(c1);
        l_r1 = iLab_idx2nameRotation(r1);
        l_r2 = iLab_idx2nameRotation(r2);
        transform = [l_c separator l_r1 separator l_r2];        
    elseif c1 ~= c2 && r1 == r2
        l_c1 = iLab_idx2nameCamera(c1);
        l_c2 = iLab_idx2nameCamera(c2);
        l_r = iLab_idx2nameRotation(r1);
        transform = [l_c1 separator l_c2 separator l_r];
    end
    
    varargout = {class, transform};
        
 

end