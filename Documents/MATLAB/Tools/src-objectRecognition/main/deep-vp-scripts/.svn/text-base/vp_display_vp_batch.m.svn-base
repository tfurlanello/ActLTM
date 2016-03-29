function vp_display_vp_batch(imdb, labels_gt, labels_pred, map2x, map2y)


    % show images
    testImgIdx   =  (imdb.images.set == 3);
    testImgNames =  imdb.images.name(testImgIdx);
    testImgDir   =  imdb.imageDir;
    
    nTest = numel(testImgNames);
    
    assert(nTest == numel(labels_pred));
    for i=1:nTest
        
        if labels_gt(i) ~= 225
            continue;
        end
        
        im = imread(fullfile(testImgDir, testImgNames{i}));
        imshow(im);
        hold on;
        
        xpred = map2x(labels_pred(i));
        ypred =  map2y(labels_pred(i));
        
        
        xgt = map2x(labels_gt(i));
        ygt =  map2y(labels_gt(i));
        
        
        scatter(xpred,ypred,80, 'MarkerEdgeColor','red', 'linewidth', 5);
        scatter(xgt,ygt,80, 'MarkerEdgeColor','green', 'linewidth', 5);
        w = waitforbuttonpress;
            
        
    end
    
end