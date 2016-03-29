function rgbd_fillImg(args)
    opts = rgbd_validateImgFilePara;
    opts = vl_argparse(opts, args);
    
    % read image and mask
    im = rgbd_readimg(opts);
    mask = rgbd_readmask(opts);
    
    if isempty(im) || isempty(mask)
        return;
    end
    
    [h,w,~] = size(im);
    
    % fill the image, such that it is square-shaped
    % get the background pixels, and use median filter to get the median
    % pixel, and at last, use this median pixel to pad the image, such that
    % it is square-shaped.    
    
    bk_mask = ~mask;
    imr = im(:,:,1); img = im(:,:,2); imb = im(:,:,3);  
    fillr = median(imr(bk_mask));
    fillg = median(img(bk_mask));
    fillb = median(imb(bk_mask));
    
    s = max(h,w);
    tar_img = uint8(repmat(reshape([fillr fillg fillb],1,1,3), s,s,1));
    
    
    if h>w
        margin = floor((s-w)/2);
        tar_img(:,margin:margin+w-1,:) = im;        
    elseif h<w
        margin = floor((s-h)/2);
        tar_img(margin:margin+h-1,:,:) = im;
    else
        tar_img = im;
    end
    
end

