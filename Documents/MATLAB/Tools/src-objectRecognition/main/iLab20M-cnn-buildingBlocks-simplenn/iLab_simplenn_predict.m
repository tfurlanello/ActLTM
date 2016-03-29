function predIndex = iLab_simplenn_predict(net, im)

    im = double(im);
    szRef = net.normalization.imageSize;
    avImg = net.normalization.averageImage;
    im = iLab_imrescale(im, szRef);
    
    if ndims(im) == 3
        avImg = reshape(avImg, [1 1 3]);
        im = im - repmat(avImg, [size(im,1) size(im,2) 1]);
    else
        im = im - avImg;
    end
    
    % evaluate CNN
  	res = vl_simplenn(net, im, [], [], ...
                      'accumulate', false, ...
                      'disableDropout', true, ...
                      'conserveMemory', false, ...
                      'sync', false, ...
                      'cudnn', false) ;

    % get label
    predictions = gather(res(end-1).x) ;
    [~,predictions] = sort(predictions, 3, 'descend') ;
    predIndex = squeeze(predictions(:,:,1,:));
 
end