saveDir = '/lab/jiaping/papers/dissertation2015/Qual';
fullImgDir = '/lab/mega/iLab-20M';


backgrounds = [10;11;13;21;33;40;59;62;68;83;89;97;107;111;112;117;119;121];
nBackgrounds = numel(backgrounds);

rng('shuffle');
rorder = randperm(nBackgrounds);
backgrounds = backgrounds(rorder);

imgs = [];
for i=1:4
    
    
    im = iLab_readimg('class', 'van', 'instance', 5, 'camera', 1, ...
                            'background', backgrounds(i), 'rotation', 2);
    
    imgs = [imgs im(:)];
end


im = imCollage(imgs, [size(im,1) size(im,2)], [2 2]);
figure; imshow(im);
imwrite(im, fullfile(saveDir, 'fig-diff-backgrounds-c1.png'));


[cameras, rotations] = iLab_getNeighboringCR(2,2, {'camera',0, 'rotation', 4});
iLab_readimgBatch(cameras, rotations, 'class', 'van', 'instance', 5, 'background', 97);