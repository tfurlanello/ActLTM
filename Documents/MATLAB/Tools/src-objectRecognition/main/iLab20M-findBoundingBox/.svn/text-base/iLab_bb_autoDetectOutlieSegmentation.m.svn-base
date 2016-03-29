function [outlieImgs, outlieCollage] = iLab_bb_autoDetectOutlieSegmentation(args)


%     resDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/iLab20M-boundingbox/res';
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
       
    factor = 2;
    %% find outlie images
    bbs_mat = cell2mat(bbs);
    potentialW = bbs_mat(:, end-1);
    potentialH = bbs_mat(:, end);
    
    idx_w = find(potentialW > (mean(potentialW) + factor*std(potentialW)));
    idx_h = find(potentialH > (mean(potentialH) + factor*std(potentialH)));
    
    idx = [idx_w(:); idx_h(:)];
    
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
    
    
    factor = 2;
    %% find outlie collage
    bb_mat = cell2mat(bb);
    potentialW = bb_mat(:,end-1);
    potentialH = bb_mat(:,end);


    idx_w = find(potentialW > (mean(potentialW) + factor*std(potentialW)));
    idx_h = find(potentialH > (mean(potentialH) + factor*std(potentialH)));
    
    idx = [idx_w(:); idx_h(:)];
    fsep = '-';
    outlieCollage = {};
    for i=1:numel(idx)
        tmp = bb_mat(idx(i),:);
        opts.camera = tmp(1);
        opts.rotation = tmp(2);
        
        collageFileName = [current_class, fsep, iLab_idx2nameInstance(current_instance), ...
                    fsep, iLab_idx2nameCamera(opts.camera), fsep, iLab_idx2nameRotation(opts.rotation) '.jpg'];
        
        outlieCollage = cat(1, outlieCollage, collageFileName);
    end


    
end