function [outlieImgs, outlieCollage] = iLab_bb_autoDiagnoseOutlieSegmentation(args)
% the difference with iLab_autoDetectOutlieSegmentation
% 1. we assume the segmentation under camera 2 3 4 5 6 7 8 (7 middle cameras) are mostly correct!
% 2. we use the segmentation statistics from 7 middle cameras to test
%    wheather segmentation from the other 4 cameras are correct.
    global pathBB;
    resDir = pathBB.round1.res;
    imgPara = iLab_validateImgFilePara;
    opts = vl_argparse(imgPara, args);
    
    current_class    = opts.class;
    current_instance = opts.instance;
    instances = iLab_retrieveInstances({'class', current_class});
    
    if ~ismember(current_instance, instances)
%         error('the instance is not in the class\n');
        outlieImgs      = {};
        outlieCollage   = {};
        return; 
    end

    cameras   = iLab_getCameras;
    rotations = iLab_getRotations;
    
    fsep = '-';
    bbs = cell(numel(cameras)*numel(rotations),1);
 	bb = cell(numel(cameras)*numel(rotations),1);

    cnt = 0;
    cnt_segs_cameras = zeros(1, numel(cameras));
    for c=1:numel(cameras)
        opts.camera = cameras(c);
        for r=1:numel(rotations)
            opts.rotation = rotations(r);
            BBsfile  = [current_class, fsep, iLab_idx2nameInstance(current_instance), ...
                fsep, iLab_idx2nameCamera(opts.camera), fsep, iLab_idx2nameRotation(opts.rotation) '.bbs'];
            BBfile  = [current_class, fsep, iLab_idx2nameInstance(current_instance), ...
                fsep, iLab_idx2nameCamera(opts.camera), fsep, iLab_idx2nameRotation(opts.rotation) '.bb'];
           
            if exist(fullfile(resDir, BBsfile), 'file') && exist(fullfile(resDir, BBfile), 'file')
                cnt = cnt + 1;
                bbs_info = dlmread(fullfile(resDir, BBsfile));   
                cnt_segs_cameras(c) = cnt_segs_cameras(c) + size(bbs_info,1);
                bbs{cnt} = [ repmat([opts.camera opts.rotation], size(bbs_info,1), 1) bbs_info];            
                bb{cnt} = [opts.camera opts.rotation dlmread(fullfile(resDir, BBfile))];
            end
        end        
    end
    
%     assert(cnt == numel(cameras)*numel(rotations));

    if cnt ~= numel(cameras)*numel(rotations)
        outlieImgs      = {};
        outlieCollage   = {};
        return; 
    end
    
    tmp_cnt = cumsum(cnt_segs_cameras);
    % use segmentation under camera 3-9 as reference
    idx_refseg = (tmp_cnt(2)+1):tmp_cnt(9);    
    
    factor = 2;
    %% find outlie images
    bbs_mat = cell2mat(bbs);
    potentialW = bbs_mat(:, end-1);
    
    refW = potentialW(idx_refseg);
    refW_mean = mean(refW);
    refW_std = std(refW);
    
    idx_w = find(potentialW > (refW_mean + factor*refW_std));    
    idx = idx_w(:); 
    
    outlieImgs = {};
    for i=1:numel(idx)
        tmp = bbs_mat(idx(i),:);
        opts.camera = tmp(1);
        opts.rotation = tmp(2);
        opts.background = tmp(3);
        opts.light = tmp(4);
        imgFileName = iLab_genImgFileName(opts);
        outlieImgs = cat(1, outlieImgs, imgFileName);
    end
    
    
    % get the unique camera-rotation pairs     
    camera_rotation = zeros(numel(outlieImgs),2);
    for i=1:numel(outlieImgs)
        opts = iLab_parseFileName(outlieImgs{i});
        camera_rotation(i,:) = [opts.camera opts.rotation];
    end
    camera_rotation = unique(camera_rotation, 'rows');
    
    
    %% find outlie collage    
    fsep = '-';
    outlieCollage = {};
    for i=1:size(camera_rotation,1)        
        opts.camera = camera_rotation(i,1);
        opts.rotation = camera_rotation(i,2);
        
        collageFileName = [current_class, fsep, iLab_idx2nameInstance(current_instance), ...
                    fsep, iLab_idx2nameCamera(opts.camera), fsep, iLab_idx2nameRotation(opts.rotation) '.jpg'];
        
        outlieCollage = cat(1, outlieCollage, collageFileName);
    end
    
  


    
end