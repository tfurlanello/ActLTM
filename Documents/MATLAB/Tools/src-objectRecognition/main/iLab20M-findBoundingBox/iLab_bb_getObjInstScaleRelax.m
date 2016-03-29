function [w,h] = iLab_bb_getObjInstScaleRelax(args)
% 
    global pathBB;
    resDir = pathBB.round1.res;
	resDir2 = pathBB.round3.res;
    error_lists = textread(fullfile(pathBB.round1.outlies, 'lists.collage'), '%s\n');
    
    imgPara = iLab_validateImgFilePara;
    opts = vl_argparse(imgPara, args);
    
    current_class    = opts.class;
    current_instance = opts.instance;
    instances = iLab_retrieveInstances({'class', current_class});
    
    if ~ismember(current_instance, instances)
        % error('the instance is not in the class\n');
        bb = [];        
        return; 
    end

    cameras   = iLab_getCameras;
    rotations = iLab_getRotations;
    
    fsep = '-';
    bb = cell(numel(cameras)*numel(rotations),1);
    cnt = 0;
    for c=1:numel(cameras)
        opts.camera = cameras(c);
        for r=1:numel(rotations)
            opts.rotation = rotations(r);
            BBfile  = [current_class, fsep, iLab_idx2nameInstance(current_instance), ...
                fsep, iLab_idx2nameCamera(opts.camera), fsep, iLab_idx2nameRotation(opts.rotation)];
            
            
            if all(cellfun('isempty', strfind(error_lists, [BBfile '.jpg'])) )
                if exist(fullfile(resDir, [BBfile '.bb']), 'file')
                    cnt = cnt + 1;
                    bb{cnt} = dlmread(fullfile(resDir, [BBfile '.bb']));   
                end
            else
                if exist(fullfile(resDir2, [BBfile '.bb']), 'file')
                    cnt = cnt  + 1;
                    bb{cnt} = dlmread(fullfile(resDir2, [BBfile '.bb']));   
                end
            end            
        end        
    end
    
    if all(cellfun('isempty', bb))
        w = [];
        h = []; 
        return;
    end
    
%     assert(cnt == numel(cameras)*numel(rotations));
    
    bb_mat = cell2mat(bb);
    potentialW = bb_mat(:,3);
    potentialH = bb_mat(:,4);
    
    
    bflagW = (potentialW < (mean(potentialW) + 3*std(potentialW)));
    bflagH = (potentialH < (mean(potentialH) + 3*std(potentialH)));
    
    ws = potentialW(bflagW);
    hs = potentialH(bflagH);
    
    w = max(ws);
    h = max(hs);    
    
    
%     ws_sorted = sort(ws);
%     hs_sorted = sort(hs);
%     diff_ws = diff(sort(ws));
%     diff_hs = diff(sort(hs));
%     
%     [~, idx_maxw] = max(diff_ws);
%     [~, idx_maxh] = max(diff_hs);
%     
%     ratio = 0.9;
% 
%     % width
%     if idx_maxw < round(ratio*cnt)
%         w = max(ws);        
%     else
%         w = ws_sorted(idx_maxw);        
%     end
%     
%     % height
%     if idx_maxh < round(ratio*cnt)
%         h = max(hs);
%     else
%         h = hs_sorted(idx_maxh);
%     end
    
    
    
    
    
end