function bb = iLab_bb_calBB(opts)
    
    saveDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/iLab20M-boundingbox';
    imgExt = '.ppm';
    
    [im, imNames] = iLab_readimg(opts);
    
    opts = iLab_parseImgName(imNames);
    
    subsaveDir = fullfile(saveDir, opts.class, ['instance-' num2str(opts.instance)]);
    if ~exist(subsaveDir, 'dir')
        mkdir(subsaveDir);
    end
     
    imppm = fullfile(subsaveDir, [imNames(1:end-4) imgExt]);
    imwrite(im, imppm);
    
    segexe = '/lab/jiaping/codem/image-segmentation/pedro/segment/segment ';
    segpara = {'0.5 300 300 '; '0.5 500 300 '; '0.5 700 300 '; '0.5 900 300 '};
    segFile = imppm;
    segout = fullfile(subsaveDir, [imNames(1:end-4) '-out' imgExt]);
    
    
    nSettings = numel(segpara);
    BBs = cell(nSettings,1);
	objCenter = iLab_getCameraCenter(opts.camera + 1);

    
    for s=1:nSettings
        segcmd = [segexe, ' ', segpara{s}, ' ', segFile, ' ', segout];
        unix(segcmd);

        imseg = imread(segout);
        imseg_gray = rgb2gray(imseg);

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
            bb =  iLab_bb_conncomp2objBB(fgmask, objCenter);
        end
        BBs{s} = bb;
    
    end
    
    bflag = cellfun(@isempty, BBs);
    BBs = BBs(~bflag);
    
    if sum(~bflag) == 0
        bb = []; return;
    end
    
    
    segpara = segpara(~bflag);
    
    centers = zeros(numel(BBs),2);    
    for i=1:numel(BBs)
        centers(i,:) = [BBs{i}(1) + BBs{i}(3)/2, BBs{i}(2) + BBs{i}(4)/2];
    end
    
    d2 = dist2(objCenter, centers);
    [~, idx] = min(d2);
    
    bb = BBs{idx};
%     segcmd = [segexe, ' ', segpara{idx}, ' ', segFile, ' ', segout];
%     unix(segcmd);
    
    
%     figure; imshow(imppm);
%     hold on; rectangle('Position',bb,'EdgeColor','g');

end