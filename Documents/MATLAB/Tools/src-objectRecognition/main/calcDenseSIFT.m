function [descriptors, fs] = calcDenseSIFT(im, scale, stride)
    
    h = size(im,1);
    w = size(im,2);
    I = single(rgb2gray(im)) ;

    x = (2*stride):stride:(w-2*stride);
    y = (2*stride):stride:(h-2*stride); 
%     
%     descriptors = zeros(128, length(x)*length(y));
%     fs = zeros(4, length(x)*length(y));
%     
    descriptors = [];
    fs = [];

    leny = length(y);
    for i=1:length(y)
%         cnt = (i-1)*length(x);
        [i, leny]
        tmpDescriptors = zeros(128, length(x));
        tmpFs = zeros(4, length(x));
        parfor j=1:length(x)           
            fc = [y(i);x(j);scale;0] ;
            [f,d] = vl_sift(I,'frames',fc,'orientations') ;
            if isempty(f) || isempty(d)
                continue;
            end
            tmpDescriptors(:,j)  = d(:,1);
            tmpFs(:,j) = f(:,1);
        end
        descriptors = cat(2, descriptors, tmpDescriptors);
        fs = cat(2, fs, tmpFs);
    end
end