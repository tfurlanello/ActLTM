function iLab_bb_calObjBB_2ndround(args, error_lists, img_blacklists, imscale, thres_mFilter)
    % inputs: which object, and which instance
    global pathBB;
    narginchk(3,5);
    if ~exist('thres_mFilter', 'var') || isempty(thres_mFilter)
        thres_mFilter = 0.60; %0.45; (default)
    end

    if ~exist('imscale', 'var') || isempty(imscale)
        imscale = 0.5;
    end
    
	opts = iLab_validateImgFilePara;    
	[opts, ~] = vl_argparse(opts, args);
    
    
%     saveDir     = pathBB.round2.objects;    
%     saveresDir  = pathBB.round2.res;
% 	oldResDir   = pathBB.round1.res;
    saveDir     = pathBB.round3.objects;    
    saveresDir  = pathBB.round3.res;
	oldResDir   = pathBB.round1.res;



	subsaveDir = fullfile(saveDir, opts.class, ['instance-' num2str(opts.instance)]);
    if ~exist(subsaveDir, 'dir')
        mkdir(subsaveDir);
    end    
    
    current_class     =  opts.class;
    current_instance  =  opts.instance;
    
	instances = iLab_retrieveInstances({'class', current_class});
    
    if sum(instances == current_instance) ~= 1
        return;
    end
    
    cameras     = iLab_getCameras;
    rotations   = iLab_getRotations;
    backgrounds = [1 2 3 4 5 6];
    lights      = iLab_getLights;
    
    ncameras     = numel(cameras);
    nrotations   = numel(rotations);
    nbackgrounds = numel(backgrounds);
    nlights = numel(lights);
    
    
    fsep = '-';
    
    for c=1:ncameras
        for r=1:nrotations
            
            opts.camera = cameras(c);
            opts.rotation = rotations(r);
            
            saveFileNameBB = [current_class, fsep, iLab_idx2nameInstance(current_instance), ...
                    fsep, iLab_idx2nameCamera(opts.camera), fsep, iLab_idx2nameRotation(opts.rotation)];
            
            if all(cellfun('isempty', strfind(error_lists, [saveFileNameBB '.jpg'])) )
                continue;
            end
                
%              if exist(fullfile(subsaveDir, [saveFileNameBB '.flag']), 'file')
%                  continue;
%              else
%                  fid = fopen(fullfile(subsaveDir, [saveFileNameBB '.flag']), 'w');
%                  fprintf(fid, '1');
%                  fclose(fid);                
%              end
                
            %% calculate bounding box under 7 different backgrounds
            tic
            BBs = cell(nbackgrounds*nlights,1);
            bkg_light_setting = zeros(nbackgrounds*nlights,2);
            
            for b=1:nbackgrounds
                for l=2:nlights
                    opts.background = backgrounds(b);
                    opts.light = lights(l);
                    
                    
                    imgFileName = iLab_genImgFileName(opts);
                    if any(~cellfun('isempty', strfind(img_blacklists, imgFileName)))
                        continue;
                    end
                    
                    bImg = iLab_imgExist(opts);
                    if bImg == false
                        BBs{(b-1)*nlights+l} = [];
                        continue;
                    else

           %                  BBs{(b-1)*nlights+l} = iLab_calBB(opts);
                    BBs{(b-1)*nlights+l} = iLab_bb_calBB2(opts, imscale); % scaling the image                    
                    bkg_light_setting((b-1)*nlights+l,:) = [opts.background opts.light];
                    
                    end            

                end
            end
            
            toc
            tmpidx              =   cellfun(@isempty, BBs);
            BBs                 =   BBs(~tmpidx);
            bkg_light_setting   =   bkg_light_setting(~tmpidx,:);            
            % bounding boxes under different backgrounds and lights 
            BBs_info            =   [bkg_light_setting cell2mat(BBs)]; 
            
            BBs_filename = fullfile(saveresDir, [saveFileNameBB '.bbs']);
            dlmwrite(BBs_filename, BBs_info, 'delimiter', ' ');
            %% get the median bounding box
             medianbb =  iLab_bb_medianBB2(BBs, thres_mFilter);
%              save(fullfile(subsaveDir, [saveFileNameBB '.mat']), 'medianbb');
             
%              fid = fopen(fullfile(subsaveDir, [saveFileNameBB '.txt']), 'w');
%              fprintf(fid, '%d %d %d %d\n', medianbb(1), medianbb(2), medianbb(3), medianbb(4));
%              fclose(fid);
             
             fid = fopen(fullfile(saveresDir, [saveFileNameBB '.bb']), 'w');
             fprintf(fid, '%d %d %d %d\n', medianbb(1), medianbb(2), medianbb(3), medianbb(4));
             fclose(fid);
             
            crops = [];            
            for b=1:nbackgrounds
                for l=2:nlights
                    opts.background = backgrounds(b);
                    opts.light = lights(l);
                    bImg = iLab_imgExist(opts);
                    if bImg == false
                        continue;
                    else
                        im = iLab_readimg(opts);
                        xs = medianbb(1);
                        xe = medianbb(1) + medianbb(3);
                        ys = medianbb(2);
                        ye = medianbb(2) + medianbb(4);
                        crop = im(ys:ye, xs:xe, :);
                        crops = [crops crop(:)];
                    end          

                end
            end  
            
            if ~isempty(crops)
                tmp = imCollage(crops, [medianbb(4)+1 medianbb(3)+1]);
%                 imwrite(tmp,fullfile(subsaveDir, [saveFileNameBB '.png']));
                imwrite(tmp,fullfile(saveresDir, [saveFileNameBB '.jpg']));
                
                im = imread(fullfile(oldResDir, [saveFileNameBB '.jpg']));
                imwrite(im, fullfile(saveresDir, [saveFileNameBB '-old.jpg']));

            end            
            
        end
    end
    
    
    
end