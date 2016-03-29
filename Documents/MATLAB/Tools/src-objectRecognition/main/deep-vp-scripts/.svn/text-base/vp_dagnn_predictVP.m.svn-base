
    impath = '/home/igpu3';
    impath = '/home2/u/kai/deep_vp/deep_vp_konsequad/KONS';

    imKONS = dir(fullfile(impath, '*.jpg'));

    % show images
    imdb = load('/home2/u/kai/deep_vp/results/google_dataset-grey-400K/vp-alexnet-dagnn-obj/imdb.mat');
    
    testImgIdx   =  (imdb.images.set == 3);
    testImgNames =  imdb.images.name(testImgIdx);
    testImgDir   =  imdb.imageDir;

    nimtest = numel(testImgNames);
    
    rng('shuffle');
    rorder = randperm(nimtest);
    
    nshow = 200;
    nshowidx = rorder(1:nshow);
% 
%     nshow = numel(imKONS);
%     imKONS = imKONS(randperm(nshow));
    for i=1:nshow

%         imfile = fullfile(impath, imKONS(i).name);
       imfile = fullfile(testImgDir, testImgNames{nshowidx(i)});

       if exist(imfile, 'file')
          im = imread(imfile);      

          pred = vp_dagnn_predictShowVP([],im);
%           title(imname);
           set(gcf, 'Units', 'normalized', 'Position', [0,0,1,1]);
          w = waitforbuttonpress;

       end

    end
    
    
% 
% for i=1:100
%    imname = ['r' num2str(i) '.jpeg'];
%    
%    imfile = fullfile(impath, imname);
%    
%    if exist(imfile, 'file')
%       im = imread(imfile);      
%       
%       pred = vp_dagnn_predictShowVP([],im);
%       title(imname);
%       w = waitforbuttonpress;
%       
%    end
%     
% end
