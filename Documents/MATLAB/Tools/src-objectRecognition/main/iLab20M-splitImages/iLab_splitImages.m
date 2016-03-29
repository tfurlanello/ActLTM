function varargout = iLab_splitImages(im)
% this function is used to split a single huge image
    
    cameras     =   0:10;
    rotations   =   0:7;
    lights      =   0:4;
    focus       =   0:2;

    w = 44;
    h = 30;
    seW = 256;
    seH = 256;
    
    imW = size(im,2);
    imH = size(im,1);
    
    if imW ~= w*seW || imH ~= h*seH
        error('The input images should be stitched images returned by Itti''s code\n');
    end
    
    seImgs = cell(w*h,1);
    nameImgs = cell(w*h,1);
    
%     curDir = pwd;
%     saveDir = fullfile(curDir, 'splitImgs');
%     if ~exist(saveDir, 'dir')
%         mkdir(saveDir);
%     end
    
	curW = 0;
    curH = 1;
    cnt = 0;
    for cam =1:11    
        for rot=1:8
            for lig=1:5
                for foc=1:3
                    cnt = cnt + 1;
                    curW = curW + 1;
                    
                    startY = (curH-1)*seH+1;
                    endY = curH*seH;
                    
                    startX = (curW-1)*seW+1;
                    endX = curW*seW;
                    
                    seImgs{cnt} = im(startY:endY, startX:endX, :);
                    nameImgs{cnt} = iLab_genImgFileName({'camera', cameras(cam), ...
                                                  'rotation', rotations(rot), ...
                                                  'light', lights(lig), ...
                                                  'focus', focus(foc)});
                    
                    if rem(curW,w) == 0
                        curH = curH + 1;
                        curW = 0;
                    end
                    
%                     imwrite(seImgs{cnt}, fullfile(saveDir,  nameImgs{cnt} ));

                end
            end
        end
    end

    varargout = {seImgs, nameImgs};
    

end