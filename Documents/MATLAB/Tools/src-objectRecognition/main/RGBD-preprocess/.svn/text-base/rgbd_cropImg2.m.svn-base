function [crop, cropbox] = rgbd_cropImg2(args, objCenter, winsize)
    
    opts = rgbd_validateImgFilePara;
    opts = vl_argparse(opts,args);
    
    im = rgbd_readimg(opts);
    
    % intelligent clipper, such that we always get a cropped image with
    % fixed winsize
    % object center: in the format of [x,y], instead of traditional y,x !
    [crop, ~, cropbox] = iLab_bb_intelligentClipper(im, winsize, objCenter);
   
end