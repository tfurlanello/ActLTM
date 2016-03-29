function imgName =  iLab_genImgFileName(args)
    % a typical image file name
    % car-i0100-b0025-c01-r04-l0-f0-crop256x256.png
    
    imgParam = iLab_validateImgFilePara;
    imgParam = vl_argparse(imgParam, args);
    
    nameClass       =   imgParam.class;
    nameInstance    =   iLab_idx2nameInstance(imgParam.instance);
    nameBackground  =   iLab_idx2nameBackground(imgParam.background);
    nameCamera      =   iLab_idx2nameCamera(imgParam.camera);
    nameRotation    =   iLab_idx2nameRotation(imgParam.rotation);
    nameLight       =   iLab_idx2nameLight(imgParam.light);
    nameFocus       =   iLab_idx2nameFocus(imgParam.focus);
    
    separator   = '-';
    imgName     = [nameClass separator nameInstance separator nameBackground separator ...
            nameCamera separator nameRotation separator nameLight separator ...
            nameFocus separator 'crop256x256.png']; %'.png']; 
    
end