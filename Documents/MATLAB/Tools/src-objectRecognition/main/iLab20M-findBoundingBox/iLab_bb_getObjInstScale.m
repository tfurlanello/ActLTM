function [wmed, hmed, wmax, hmax] = iLab_bb_getObjInstScale(args)
% inputs
%       arg - a cell or struct array, contain 'class' and 'instance'
% outputs
%       winsize - clipping window size

    global pathBB;
    resDir  = pathBB.round1.res;
	resDir2 = pathBB.round3.res;
    error_lists = textread(fullfile(pathBB.round1.outlies, 'lists.collage'), '%s\n');

    imgPara = iLab_validateImgFilePara;
    opts = vl_argparse(imgPara, args);
    
    current_class    = opts.class;
    current_instance = opts.instance;
    instances = iLab_retrieveInstances({'class', current_class});
    
    if ~ismember(current_instance, instances)
        % error('the instance is not in the class\n');
        wmed = []; hmed = [];  
        wmax = []; hmax = [];
        return; 
    end

    cameras   = iLab_getCameras;
    rotations = iLab_getRotations;
    
    fsep = '-';
    bbs = cell(numel(cameras)*numel(rotations),1);
    cnt = 0;
    for c=1:numel(cameras)
        opts.camera = cameras(c);
        for r=1:numel(rotations)
            opts.rotation = rotations(r);
            BBfile  = [current_class, fsep, iLab_idx2nameInstance(current_instance), ...
                fsep, iLab_idx2nameCamera(opts.camera), fsep, iLab_idx2nameRotation(opts.rotation)];
            
            if all(cellfun('isempty', strfind(error_lists, [BBfile '.jpg'])) )
                if exist(fullfile(resDir, [BBfile '.bbs']), 'file')
                    cnt = cnt + 1;
                    bbs{cnt} = dlmread(fullfile(resDir, [BBfile '.bbs']));   
                end
            else
                if exist(fullfile(resDir2, [BBfile '.bbs']), 'file')
                    cnt = cnt + 1;
                    bbs{cnt} = dlmread(fullfile(resDir2, [BBfile '.bbs']));   
                end
            end            
            
        end        
    end
    
        
    if all(cellfun('isempty', bbs))
        wmed = [];
        hmed = [];
        wmax = [];
        hmax = [];
        return;
    end
%     assert(cnt == numel(cameras)*numel(rotations));
    
    widths  = zeros(cnt,1);
    heights = zeros(cnt,1);
    for i=1:cnt
        bbs_info = bbs{i};
        [widths(i), heights(i)] = iLab_bb_medianBBs2wh(bbs_info(:,1), ...
                        bbs_info(:,2), bbs_info(:,3:end));
    end
    
    wmed = max(widths);
    hmed = max(heights);
    
    %% first prune the outlies and then get the max from the inlies
    bbs_mat = cell2mat(bbs);
    potentialW = bbs_mat(:, end-1);
    potentialH = bbs_mat(:, end);
    
    bflagW = (potentialW < (mean(potentialW) + 3*std(potentialW)));
    bflagH = (potentialH < (mean(potentialH) + 3*std(potentialH)));
    
    ws = potentialW(bflagW);
    hs = potentialH(bflagH);
    
    wmax = max(ws);
    hmax = max(hs);
    
    
%     wmax = max(bbs_mat(:,end-1));
%     hmax = max(bbs_mat(:,end));
    
    
end