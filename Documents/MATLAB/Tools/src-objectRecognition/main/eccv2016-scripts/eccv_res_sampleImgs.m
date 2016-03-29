
resSaveDir = '/lab/jiaping/papers/ECCV2016/results';


[cameras, rotations] = iLab_getNeighboringCR(2,2, {'camera',0, 'rotation', 4});
[im, ims] = iLab_readimgBatch(cameras, rotations, 'class', 'van', 'instance', 5, 'background', 97);


imwrite(im, fullfile(resSaveDir, 's-collage.png'));

nImgs = size(ims,2);

for i=1:nImgs
    i_im = ims(:,i);
    i_im = reshape(i_im, [256 256 3]);
    imwrite(i_im, fullfile(resSaveDir, ['s-' num2str(i) '.png']));
end