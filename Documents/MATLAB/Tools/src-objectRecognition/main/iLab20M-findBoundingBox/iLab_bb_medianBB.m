function medianbb =  iLab_bb_medianBB(bbs, thres)
% this function is designed specifically for iLab dataset
% compute the median bounding box
% bounding box has to be overlapped by 80%

    narginchk(1,2);
    if ~exist('thres','var') || isempty(thres)
        thres = 0.7;
    end

    nBBs = numel(bbs);

    overlaps = zeros(nBBs, nBBs);
    
    for i=1:nBBs
        ibb = bbs{i};
        iminx = ibb(1);
        imaxx = ibb(1) + ibb(3);
        
        iminy = ibb(2);
        imaxy = ibb(2) + ibb(4);
        for j=1:nBBs
            if j==i
                continue;
            end
            jbb = bbs{j};
            
            jminx = jbb(1);
            jmaxx = jbb(1) + jbb(3);

            jminy = jbb(2);
            jmaxy = jbb(2) + jbb(4);
            
            minx = min([iminx jminx]);
            miny = min([iminy jminy]);
            
            maxx = max([imaxx jmaxx]);
            maxy = max([imaxy jmaxy]);
            
            ijbb = [minx miny (maxx-minx) (maxy-miny)];
            
            area = rectint(ibb, ijbb);
            overlaps(i,j) =  double(area) / (ijbb(3) * ijbb(4));
        end
    end
    flag = overlaps > thres;
    [I,J] = ind2sub(size(overlaps), find(flag));
    
    if isempty(I) || isempty(J)
        medianbb = [];
        return;
    end
    
    nPairs = numel(I);
    bflag = zeros(nPairs, 1) > 1;
    
    for i=1:nPairs
        ref = [I(i) J(i)];
        tar = [J(i) I(i)];
        
        res = ([I J] == repmat(tar, nPairs,1));
        
        if sum(sum(res,2) == 2) == 1
            bflag(i) = true;
        end
        
    end
    
    if sum(bflag) == 0
        medianbb = [];
        return;
    end
    
    I = sort(I(bflag));
    J = sort(J(bflag));
    
    bbs_filtered = bbs(I);    
    % get the maximum bounding box of bbs
    
    minx = 0;
    miny = 0;
    maxx = 0;
    maxy = 0;
    
    nBBs_filtered = numel(bbs_filtered);
    
    for i=1:nBBs_filtered
        ibb = bbs_filtered{i};
        if i==1
            minx = ibb(1);
            miny = ibb(2);
            maxx = ibb(1) + ibb(3);
            maxy = ibb(2) + ibb(4);
            
        else
            minx = min([minx ibb(1)]);            
            miny = min([miny ibb(2)]);
            maxx = max([maxx ibb(1) + ibb(3)]);
            maxy = max([maxy ibb(2) + ibb(4)]);
            
        end
        
        
    end
    
    
    medianbb = [minx miny maxx maxy];


end