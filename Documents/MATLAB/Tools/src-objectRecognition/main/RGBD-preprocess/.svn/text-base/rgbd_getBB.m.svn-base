function cropbox = rgbd_getBB(args)
    
    opts = rgbd_validateImgFilePara;
    opts = vl_argparse(opts,args);
    
    immask = rgbd_readmask(opts, 'raw');
    if isempty(immask)
        cropbox = [];
        return;
    end
    
    CC = bwconncomp(immask);    
    assert(numel(CC) == 1);    
    pixelIdxList = CC(1).PixelIdxList;
    
    [I,J] = ind2sub(size(immask), pixelIdxList{1});    
    miny = min(I); maxy = max(I);
    minx = min(J); maxx = max(J);
    
    cropbox = [minx miny (maxx-minx+1) (maxy-miny+1)];
    
end