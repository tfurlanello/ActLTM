% visualization of BOW

% (1) add clustering path
addpath(genpath('/lab/jiaping/projects/google-glass-project/src/clustering'));

saveDir = '/lab/ilab/30/jiaping/iLab20M/dense-sift';
load('testManifoldFileInfo.mat');

classNames = testFileInfo.classNames;
nClasses = numel(classNames);
classInfo = testFileInfo.classInfo;

rng('shuffle');
rclassIdx = randperm(nClasses);

cameraStr = {'c00', 'c01', 'c02', 'c03', 'c04', ...
             'c05', 'c06', 'c07', 'c08', 'c09', 'c10'};
nCameras = numel(cameraStr);
nBackgrounds = 5;
nImgBackGround = 88;


nclusters = [50 100 200 300];
for ii=1:nClasses
    i = rclassIdx(ii);
    c_instanceInfo  = classInfo(i).instanceInfo;
    c_instanceNames = classInfo(i).instanceNames;
    nInstances      = numel(c_instanceNames);
    rinstanceIdx    = randperm(nInstances);
    for jj=1:nInstances
        j = rinstanceIdx(jj);
        j_instanceName  = c_instanceInfo(j).instanceName;
        imageFiles      = c_instanceInfo(j).instanceFiles;
        nimgs           = numel(imageFiles);
         
        imageNames = {};
        for k=1:nimgs
            slashIdx = strfind(imageFiles{k}, '/');
            imName = imageFiles{k}((slashIdx(end)+1):end);
             imageNames = cat(1, imageNames, imName );
        end
        
        close all;
        idxshows = randi(nimgs,1,2);
        idxshows = [29 205];
        img1 = imread(imageFiles{idxshows(1)});
        img2 = imread(imageFiles{idxshows(2)});
        


        [f1,d1] = vl_sift(single(rgb2gray(img1))) ;
        [f2,d2] = vl_sift(single(rgb2gray(img2))) ;
        figure; imagesc(img1);  axis equal; axis tight;
        h1 = vl_plotframe(f1) ;
        set(h1,'color','y','linewidth',3) ;
        
        figure; imagesc(img2);  axis equal; axis tight;
        h2 = vl_plotframe(f2) ;
        set(h2,'color','y','linewidth',3) ;

        fc = [74;125;3;0] ;
        fc = [195; 111;3;0] ;
        fc = [106; 37;3;0] ;

        I = single(rgb2gray(img1)) ;
        [f,d] = vl_sift(I,'frames',fc,'orientations') ;
        h3 = vl_plotsiftdescriptor(d(:,1),f(:,1)) ;
        set(h3,'color','y') ;
        
        
        fc = [54; 121;3;0] ;
        fc = [175; 117;3;0] ;
        fc = [116; 39;3;0] ;

        I = single(rgb2gray(img2)) ;
        [f,d] = vl_sift(I,'frames',fc,'orientations') ;
        h4 = vl_plotsiftdescriptor(d(:,1),f(:,1)) ;
        set(h4,'color','y') ;    
        
        %% 

        fc = [74;125;3;0] ;
        fc = [195; 111;3;0] ;
%         fc = [106; 37;3;0] ;

        I = single(rgb2gray(img1)) ;
        [f,d] = vl_sift(I,'frames',fc,'orientations') ;
        h3 = vl_plotsiftdescriptor(d(:,1),f(:,1)) ;
        set(h3,'color','g') ;
        
        
        fc = [54; 121;3;0] ;
        fc = [175; 117;3;0] ;
%         fc = [116; 39;3;0] ;

        I = single(rgb2gray(img2)) ;
        [f,d] = vl_sift(I,'frames',fc,'orientations') ;
        h4 = vl_plotsiftdescriptor(d(:,1),f(:,1)) ;
        set(h4,'color','g') ;  
        
        %%
        fc = [74;125;3;0] ;
%         fc = [195; 111;3;0] ;
%         fc = [106; 37;3;0] ;

        I = single(rgb2gray(img1)) ;
        [f,d] = vl_sift(I,'frames',fc,'orientations') ;
        h3 = vl_plotsiftdescriptor(d(:,1),f(:,1)) ;
        set(h3,'color','m') ;
        
        
        fc = [54; 121;3;0] ;
%         fc = [175; 117;3;0] ;
%         fc = [116; 39;3;0] ;

        I = single(rgb2gray(img2)) ;
        [f,d] = vl_sift(I,'frames',fc,'orientations') ;
        h4 = vl_plotsiftdescriptor(d(:,1),f(:,1)) ;
        set(h4,'color','m') ;  
    

    end
end