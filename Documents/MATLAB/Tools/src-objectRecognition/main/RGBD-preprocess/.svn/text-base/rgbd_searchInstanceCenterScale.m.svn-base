function imInfo_more = rgbd_searchInstanceCenterScale
% for each instance under the same camera, it's shot under different
% rotations, we need to use the same cropping window for all these
% differently-viewed images

    margin = 5;
    if nargin > 0
        return;
    end
    
    imInfo = rgbd_getImgInfo('raw');    
    nImages = numel(imInfo);    
    imInfo_more = imInfo;
    
    % get scale for each image, first
    cnt = 0;
    for i=1:nImages     
        if rem(i,500) == 0
            i
        end        
        i_imInfo = imInfo(i);   
        cropbox = rgbd_getBB(i_imInfo);
        if isempty(cropbox)
            continue;
        end

        cnt = cnt + 1;
        imInfo_more(cnt).center = [round(cropbox(1)+cropbox(3)/2) ...
                                        round(cropbox(2)+cropbox(4)/2)]; 
        imInfo_more(cnt).scale    = max(cropbox(3), cropbox(4));  
        imInfo_more(cnt).class    = imInfo(i).class;
        imInfo_more(cnt).instance = imInfo(i).instance;
        imInfo_more(cnt).camera   = imInfo(i).camera;
        imInfo_more(cnt).frame    = imInfo(i).frame;
    end
    
    imInfo_more = imInfo_more(1:cnt);    
    imHierarchy = rgbd_getImgHierarchy('raw');    
    % get an uniform scale for all images of the same instance under the
    % same camera, then    
    categories  =  {imInfo_more.class};
    instances   =  cell2mat({imInfo_more.instance});
    cameras     =  cell2mat({imInfo_more.camera});
    frames      =  cell2mat({imInfo_more.frame});
    scales      =  cell2mat({imInfo_more.scale});
                 
    classes     =   rgbd_getClasses;
    nclasses    =   numel(classes);
    cameras3     =   rgbd_getCameraIdx;
    ncameras    =   numel(cameras3);
    
    uni_scales = zeros(cnt,1);    
    for c=1:nclasses        
        fprintf(1, 'processing: object %s\n', classes{c});
        c_class = classes{c};        
        b_class = strcmp(c_class, categories);
        instancesID = imHierarchy.(c_class).instancesID;    
        
        for i=1:numel(instancesID)
            b_instance = instancesID(i) == instances;            
            for cam=1:ncameras
               b_camera = cameras3(cam) == cameras;               
               bflag = b_class & b_instance & b_camera;
               tmp_scales = scales(bflag);
               uni_scales(bflag) = max(tmp_scales);
            end
        end        
    end
    
    uni_scales = uni_scales + 2*margin;
    % set scales 
%     imInfo_more = setfield(imInfo_more, {1:cnt}, 'scale', {ones(1,cnt)}, uni_scales);
    for i=1:cnt
        imInfo_more(i).scale = uni_scales(i);
    end
    imInfo = imInfo_more; clear imInfo_more;    
    global workdir;    
    save(fullfile(workdir,  'main', 'RGBD-data-info',  'imgInfo-center-scale.mat'), 'imInfo');

        
end