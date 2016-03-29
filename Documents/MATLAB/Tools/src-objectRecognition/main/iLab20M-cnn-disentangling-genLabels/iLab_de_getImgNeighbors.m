function neighbors = iLab_de_getImgNeighbors(arg)

	arg = vl_argparse(iLab_validateImgFilePara, arg);
    
    c = arg.camera;
    r = arg.rotation;
    
    nei_r = iLab_de_getNeighboringRotation(r);    
    nei_cameras = iLab_de_getNeighboringCameras;    
    nei_c = nei_cameras(nei_cameras(:,1) == c,2);
    
    neighbors = {};
    if ~isempty(nei_r)
        arg1 = arg;
        arg1.rotation = nei_r;
        neighbors{end+1} = arg1;
    end
    
    if ~isempty(nei_c)
        arg2 = arg;
        arg2.camera = nei_c;
        neighbors{end+1} = arg2;
    end
    
    
end