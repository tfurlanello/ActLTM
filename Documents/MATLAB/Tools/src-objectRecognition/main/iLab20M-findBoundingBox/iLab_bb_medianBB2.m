function [medianbb, bflag] =  iLab_bb_medianBB2(bbs, thres)
% this function is designed specifically for iLab dataset
% compute the median bounding box
% bounding box has to be overlapped by 80%

    narginchk(1,2);
    if ~exist('thres','var') || isempty(thres)
        thres = 0.7;
    end
    
    nBBs = numel(bbs);

    bbs_mat = cell2mat(bbs);
    minx = bbs_mat(:,1);
    miny = bbs_mat(:,2);
    maxx = bbs_mat(:,1) + bbs_mat(:,3);
    maxy = bbs_mat(:,2) + bbs_mat(:,4);
    
    medianx1 = median(minx);
    mediany1 = median(miny);
    medianx2 = median(maxx);
    mediany2 = median(maxy);
    
    ref = [medianx1 mediany1 medianx2-medianx1 mediany2-mediany1];
    
    
    overlaps = zeros(nBBs,1);
    bflag = zeros(nBBs,1) > 1.0;
    for i=1:nBBs
        tar = bbs{i};
        
        area = rectint(tar, ref);
        
        per = area / (tar(3)*tar(4));
        overlaps(i) = per;
        if per > thres
            bflag(i) = true;
        end        
    end
    
    if sum(bflag) == 0
        medianbb = uint8(ref);
        return;
    end
    
    bbs_filtered = bbs(bflag);
    bbs_filtered = cell2mat(bbs_filtered);
    
	minx = bbs_filtered(:,1);
    miny = bbs_filtered(:,2);
    maxx = bbs_filtered(:,1) + bbs_filtered(:,3);
    maxy = bbs_filtered(:,2) + bbs_filtered(:,4);
    

  
    
    medianbb = [min(minx) min(miny) max(maxx)-min(minx) max(maxy)-min(miny)];


end