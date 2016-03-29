% extract SIFT feature points from selected images

siftFileInfo = ...
    load('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-experiments/data-sift.mat');

iLabInfo    = load('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-data-info/imagefiles-info.mat');
imageNames  = getImgFileNamesiLab();


%% selected images
bSelected       =   siftFileInfo.bSelected;

imageNames_sel  =   imageNames(bSelected);
imgDirsIdx_sel  =   iLabInfo.imgDirsIdx(bSelected);
nImages_sel     =   sum(bSelected);
labels_sel      =   siftFileInfo.labelsSelected;

% move the images to a new place
saveDir = '/lab/tmp2ig2/u/jiaping/bow-visualization';
if ~exist(fullfile(saveDir, 'selectedImages'), 'dir')
    mkdir(fullfile(saveDir, 'selectedImages'));
end

if ~exist(fullfile(saveDir, 'siftDescriptors'), 'dir')
    mkdir(fullfile(saveDir, 'siftDescriptors'));
end

intervals = linspace(1, nImages_sel, 5);
peak_thresh = 5;

for i=1:4
    
    if ~exist(fullfile(saveDir, [num2str(i) '-sel.txt']), 'file')
        fid = fopen(fullfile(saveDir, [num2str(i) '-sel.txt']), 'wt');
        fprintf(fid, '1');
        fclose(fid);
        
        fs = cell(nImages_sel,1);
        ds = cell(nImages_sel,1);
        parfor j=round(intervals(i)):round(intervals(i+1))
             im = imread(fullfile(iLabInfo.imgDirs{imgDirsIdx_sel(j)}, imageNames_sel{j}));
             imwrite(im, fullfile(saveDir, 'selectedImages',imageNames_sel{j}), 'png'); 
             
             I = single(rgb2gray(im)) ;
             [fs{j},ds{j}] = vl_sift(I, 'PeakThresh', peak_thresh) ;
        end  
        
        for j=round(intervals(i)):round(intervals(i+1))
             f = fs{j};
             d = ds{j};
             save(fullfile(saveDir, 'siftDescriptors', [imageNames_sel{j}(1:end-4) '.mat']), 'f','d');
        end 
        clear fs ds;

    end

end








