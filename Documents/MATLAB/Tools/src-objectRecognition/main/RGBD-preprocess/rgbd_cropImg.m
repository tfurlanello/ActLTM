function [crop, cropbox] = rgbd_cropImg(args)
    
    margin = 5;

    opts = rgbd_validateImgFilePara;
    opts = vl_argparse(opts,args);
    
    im = rgbd_readimg(opts);
    immask = rgbd_readmask(opts);
    [h,w] = size(immask);
    
    CC = bwconncomp(immask);    
    assert(numel(CC) == 1);    
    pixelIdxList = CC(1).PixelIdxList;
    
    [I,J] = ind2sub(size(immask), pixelIdxList{1});    
    miny = min(I); maxy = max(I);
    minx = min(J); maxx = max(J);
    
    minx = max(minx-margin,1);
    maxx = min(maxx+margin,w);
    miny = max(miny-margin,1);
    maxy = min(maxy+margin,h);
    
    
    objCenter   =   [round((minx+maxx)/2) round((miny+maxy)/2)];
    winsize     =   max(maxy-miny+1, maxx-minx+1);
    
    % intelligent clipper, such that we always get a cropped image with
    % fixed winsize
    % object center: in the format of [x,y], instead of traditional y,x !
    [crop, ~, cropbox] = iLab_bb_intelligentClipper(im, winsize, objCenter);
   
end