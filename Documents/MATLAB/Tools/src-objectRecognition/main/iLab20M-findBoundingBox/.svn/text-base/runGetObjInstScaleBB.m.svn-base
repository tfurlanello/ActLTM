global pathBB;
saveDir = pathBB.round3.bb;
[objInstNames, scales, objInstCRnames, BBs] =  iLab_bb_getObjInstScaleBB;

fscales = fopen(fullfile(saveDir, 'iLab20M-obj-scales.txt'), 'w');
fbb = fopen(fullfile(saveDir, 'iLab20M-obj-bb.txt'), 'w');

for i=1:numel(objInstNames)
    fprintf(fscales, '%s %d %d %d\n', objInstNames{i}, scales(i,1), scales(i,2), scales(i,3));
end

for i=1:numel(objInstCRnames)
    fprintf(fbb, '%s %d %d %d %d\n', objInstCRnames{i}, ...
                    BBs(i,1), BBs(i,2), BBs(i,3), BBs(i,4));
end

fclose(fscales);
fclose(fbb);