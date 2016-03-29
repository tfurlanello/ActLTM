function [bb, polygon] = iLab_bb_getObjInstMaxBB(args)
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
        bb = [];
        polygon = [];
        return;
    end
    
%     assert(cnt == numel(cameras)*numel(rotations));
    
    bb_mat = cell2mat(bb);
    potentialW = bb_mat(:,3);
    potentialH = bb_mat(:,4);
    
    
    bflagW1 = (potentialW < (mean(potentialW) + 3*std(potentialW)));
    bflagW2 = (potentialW > (mean(potentialW) - 3*std(potentialW)));
    
    bflagH1 = (potentialH < (mean(potentialH) + 3*std(potentialH)));
    bflagH2 = (potentialH > (mean(potentialH) - 3*std(potentialH)));
    
    bflagW = (bflagW1 & bflagW2);
    bflagH = (bflagH1 & bflagH2);
    
    bflag = (bflagW & bflagH);
    
    bb_mat = bb_mat(bflag,:);
    
    
    
    
%     assert(cnt == numel(cameras)*numel(rotations));
    cnt = size(bb_mat,1);
    for i=1:cnt
        if i==1
            outpoly = rectangle2polygon(bb_mat(i,:));
        else
            poly2 = rectangle2polygon(bb_mat(i,:));
            type = 3; % union;
            outpoly = PolygonClip(outpoly,poly2,type);            
        end
    end
    
    polygon = outpoly;
    bb = polygon2rectangle(outpoly);
    
end