function bb =  iLab_bb_conncomp2objBB(bImg, objCenter)

    % close image before any operation
%     se = strel('disk',5);
%     closeBW = imclose(bImg,se);

    cc = bwconncomp(bImg);
    
    numPixels = cellfun(@numel,cc.PixelIdxList);
    numObj = cc.NumObjects;
    
    subh = cell(numObj,1);
    subw = cell(numObj,1);    
    bbs = cell(numObj,1);
    flagWithin = zeros(numObj,1) > 1.0;
    
    for o=1:numObj
        [subh{o}, subw{o}] = ind2sub(cc.ImageSize, cc.PixelIdxList{o});
        minh = min(subh{o});
        maxh = max(subh{o});
        
        minw = min(subw{o});
        maxw = max(subw{o});
        
        bbs{o} = [minw minh (maxw - minw) (maxh - minh)];
        
        objx = objCenter(1);
        objy = objCenter(2);
        
        if objx >= minw && objx <= maxw && ...
                objy >= minh && objy <= maxh
            flagWithin(o) = true;
        end        
    end
    
    if sum(flagWithin) == 0
        bb = [];
        return;
    end
    
    numPixels =  numPixels(flagWithin);
    bbs       =  bbs(flagWithin);
    [~, idx] = max(numPixels);
    bb = bbs{idx};   
    

end
