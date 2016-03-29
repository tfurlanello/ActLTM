% bow encoding, and then t-SNE visualization
 

%% selected image information

siftFileInfo = ...
    load('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-experiments/data-sift.mat');
 
imageNames  = getImgFileNamesiLab(); 
bSelected       =   siftFileInfo.bSelected;
imageNames_sel  =   imageNames(bSelected);
nImages_sel     =   sum(bSelected);
labels_sel      =   siftFileInfo.labelsSelected;

%% image and sift 
imgDir = '/lab/tmp2ig2/u/jiaping/bow-visualization/selectedImages';
siftDir = '/lab/tmp2ig2/u/jiaping/bow-visualization/siftDescriptors';
saveDir = '/lab/tmp2ig2/u/jiaping/bow-visualization';

SIFTs = cell(nImages_sel,1);
imgFiles = strcat( [imgDir filesep], imageNames_sel); 
for i=1:nImages_sel
    load(fullfile(siftDir, [imageNames_sel{i}(1:end-4) '.mat']));
    SIFTs{i} = d';
end

sparsification = [20 10 5 1];
for i=1:numel(sparsification)
    if exist(fullfile(saveDir, ['sparse-' num2str(sparsification(i)) '-600']), 'file')
        continue;
    end
    fid = fopen(fullfile(saveDir, ['sparse-' num2str(sparsification(i)) '-600']), 'wt');
    fprintf(fid, '1\n');
    fclose(fid);
    
    idx = 1:sparsification(i):nImages_sel;
    i_SIFTs     =   SIFTs(idx);
    i_imgFiles  =   imgFiles(idx);
    i_sig = encoderBoW(i_SIFTs, 600);
    save(fullfile(saveDir,  ['encoder-bow-' num2str(sparsification(i)) '-600.mat']), 'i_imgFiles', 'i_SIFTs',  'i_sig');
end