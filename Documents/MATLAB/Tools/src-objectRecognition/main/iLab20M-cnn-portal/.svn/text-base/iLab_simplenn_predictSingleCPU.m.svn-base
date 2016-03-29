function pred = iLab_simplenn_predictSingleCPU(net, im)
    
    if ~exist('net','var') || isempty(net)    
        load('/lab/ilab/30/kai-vp/google_dataset/vp-alexnet-simplenn-obj/net-epoch-17.mat');
    end

    ims = imresize(im, [227 227]);
    ims = single(ims);
    imo = zeros(size(ims,1), size(ims,2), size(ims,3), 2);
    imo(:,:,:,1) = ims;
    imo(:,:,:,2) = ims;

    net.layers{end}.class = [1 1] ;

    res = vl_simplenn(net,single(imo), [], [], ...
                      'accumulate', 0, ...
                      'disableDropout', true, ...
                      'conserveMemory', 0, ...
                      'sync', 0, ...
                      'cudnn', 0) ;
                  
    predictions = gather(res(end-1).x) ;
    [~,predictions] = sort(predictions, 3, 'descend') ;
    pred = squeeze(predictions(:,:,1:5,:));         
    
    pred = pred(:,1);

end