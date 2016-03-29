function pred = vp_simplenn_predictShowVP(net, im)
    
    if ~exist('net','var') || isempty(net)    
        load('/lab/ilab/30/kai-vp/google_dataset/vp-alexnet-simplenn-obj/net-epoch-17.mat');
    end

    ims = iLab_imrescale(im, [227 227]);
    ims = single(ims);
    
    mes = net.normalization.averageImage;
%     mes = [mean(im(:,:,1)) mean(im(:,:,2)) mean(im(:,:,3))];
    ims(:,:,1) = ims(:,:,1) - mes(1);
    ims(:,:,2) = ims(:,:,2) - mes(2);
    ims(:,:,3) = ims(:,:,3) - mes(3);

     
    imo = zeros(size(ims,1), size(ims,2), size(ims,3), 2);
    imo(:,:,:,1) = ims;
    imo(:,:,:,2) = ims;

    net.layers{end}.class = [1 1] ;

    res = vl_simplenn(net,single(imo), [], [], ...
                      'accumulate', 0, ...
                      'disableDropout', false, ...
                      'conserveMemory', 0, ...
                      'sync', 0, ...
                      'cudnn', 0) ;
                  
    predictions = gather(res(end-1).x) ;
    [~,predictions] = sort(predictions, 3, 'descend') ;
    pred = squeeze(predictions(:,:,1:5,:));         
    
    pred = pred(:,1);
    
    
    load('/lab/ilab/30/kai-vp/google_dataset/vp-alexnet-simplenn-obj/mapping.mat');
    mapy = containers.Map(mapping(:,1),mapping(:,3));
    mapx = containers.Map(mapping(:,1),mapping(:,2));
    
    
    figure; 
    imshow(uint8(iLab_imrescale(im, [300 300])));
    hold on;
    
 
    colors = {'red', 'green', 'blue'};
    for i=1:3
        x = mapx(pred(i));
        y = mapy(pred(i));
        scatter(x, y ,80, 'MarkerEdgeColor', colors{i}, 'linewidth', 5);
        hold on;
    end
    

end