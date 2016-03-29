% prepare training and testing list
% in the format of deCNN


%% there are 4 cases:
%% case one: there are 5 images per class
%% case two: there are 10 images per class
%% case three: there are 20 images per class
%% case four: there are 40 images per class

nperclass = [5 10 20 40];
ncategories = 1000;
savedir = '/lab/igpu3/u/jiaping/imageNet2010/images/ECCV-metadata';

%% read training lists
trainListFile = '/lab/igpu3/u/jiaping/imageNet2010/images/train-crop-256/train.txt';
[imnames, imlabels] = textread(trainListFile, '%s %s\n');
imlabels = str2double(imlabels);

for i=1:numel(nperclass)
   
    fid = fopen(fullfile(savedir, ['train-' num2str(nperclass(i)) '.txt']), 'w');
    for c=1:ncategories
       
        bflag = imlabels == c;        
        c_imnames = imnames(bflag);   
        nimgs = min(nperclass(i), numel(c_imnames));
        c_imnames = c_imnames(1:nimgs);
        
        for k=1:nimgs           
            fprintf(fid, '%d %d %s %s\n', c, c, ...
                c_imnames{k}, c_imnames{k});
        end        
    end
    fclose(fid);
    
end

%% read test lists
testListFile = '/lab/igpu3/u/jiaping/imageNet2010/images/test-crop-256/test.txt';
[imnames, imlabels] = textread(testListFile, '%s %s\n');
imlabels = str2double(imlabels);

fid = fopen(fullfile(savedir, 'test.txt'), 'w');
for i=1:numel(imlabels)
    fprintf(fid, '%d %d %s %s\n', imlabels(i), imlabels(i), ...
                imnames{i}, imnames{i});
end
fclose(fid);
