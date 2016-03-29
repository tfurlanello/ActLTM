function bb = iLab_bb_calBB2_debug(opts, scale)
% difference from 'iLab_calBB.m'
% here we use a matlab wrapper of 'pedro 's segmentation' code to run
% potentially more efficient
    

    saveDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/iLab20M-boundingbox-tight';
    
    
    
    narginchk(1,2);
    if ~exist('scale', 'var') || isempty(scale)
        scale = 0.5;
    end
    
    [im, imNames] = iLab_readimg(opts);
    [h,w] = size(im);
    
    opts = iLab_parseImgName(imNames);
    
    subsaveDir = fullfile(saveDir, opts.class, ['instance-' num2str(opts.instance)]);
    if ~exist(subsaveDir, 'dir')
        mkdir(subsaveDir);
    end
  
    
%     segexe = '/lab/jiaping/codem/image-segmentation/pedro/segment/segment ';
%     segpara = {'0.5 300 300 '; '0.5 500 300 '; '0.5 700 300 '; '0.5 900 300 '};
%     segFile = imppm;
%     segout = fullfile(subsaveDir, [imNames(1:end-4) '-out' imgExt]);
    
    Ks        = 2*[600 1000 1400 1800]* (scale^2);
    minsize   = round(1000 * (scale^2));
    im_scaled = imresize(im, scale);    
	objCenter = iLab_getCameraCenter(opts.camera + 1);
    objCenter_scaled = objCenter * scale;
        
    nSettings = numel(Ks);
    BBs = cell(nSettings,1);
    
    for s=1:nSettings

        imseg_gray = segmentFelzenszwalb(im_scaled, 0.6, Ks(s), minsize, false);

        seglabels = unique(imseg_gray(:));
        nLabels = numel(seglabels);
        nPixels = zeros(nLabels,1);

        for i=1:nLabels
            flag = (imseg_gray == seglabels(i));
            nPixels(i) = sum(flag(:));
        end

        [~, idx] = max(nPixels);

        bgmask  =  (imseg_gray == seglabels(idx));
        fgmask    =  ~bgmask;    
        bgPercent = sum(bgmask(:)) / numel(bgmask(:));

        %% if the background is smaller than 50% percent,
        % simply drop this background

        if bgPercent < 0.5
            bb = [];
        else
            bb =  iLab_bb_conncomp2objBB(fgmask, objCenter_scaled);
        end
        BBs{s} = bb;
    
    end
    
    bflag = cellfun(@isempty, BBs);
    BBs = BBs(~bflag);    
    if sum(~bflag) == 0
        bb = []; return;
    end

    % median filter
    [~, bflag] =  iLab_bb_medianBB2(BBs, 0.85);
	BBs = BBs(bflag);    
    if sum(bflag) == 0
        bb = []; return;
    end
    
    centers = zeros(numel(BBs),2);    
    for i=1:numel(BBs)
        centers(i,:) = [BBs{i}(1) + BBs{i}(3)/2, BBs{i}(2) + BBs{i}(4)/2];
    end
    
    d2 = dist2(objCenter_scaled, centers);
    [~, idx] = min(d2);
    
    bb = BBs{idx};
    
    % make sure the bounding box is contained in the image
    bb =  bb / scale;
    bb(1:2) = floor(bb(1:2));
    bb(3:4) = ceil(bb(3:4));
    bb(1) = max(1, bb(1)); 
    bb(1) = min(w, bb(1));
    bb(2) = max(1, bb(2));
    bb(2) = min(h, bb(2));
    
    xe_bb = bb(1) + bb(3);
    ye_bb = bb(2) + bb(4);
    
    xe_bb = min(xe_bb, w);
    ye_bb = min(ye_bb, h);
    
    bb(3) = xe_bb - bb(1);
    bb(4) = ye_bb - bb(2);
    


end