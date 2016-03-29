% generate tables for iLab20M dataset

resSaveDir = '/lab/jiaping/papers/ECCV2016/results';
acc_file = fullfile(resSaveDir, 'accuracies-ImageNet.mat');

load(acc_file);


fid = fopen(fullfile(resSaveDir, 'table-ImageNet.txt'), 'w');
fprintf(fid,'  methods & 5 & 10 & 20 & 40 \\tabularnewline\n\\hline\n');

fprintf(fid, ' AlexNet (scratch) ');
for i=1:4
    fprintf(fid, ' & %.2f ', 100*acc_scratch_top5(i));
end
fprintf(fid, '\\tabularnewline\n\\hline\n');

fprintf(fid, ' AlexNet (AlexNet-iLab20M) ');
for i=1:4    
    fprintf(fid, ' & %.2f ', 100*acc_alexnet_top5(i));
end
fprintf(fid, '\\tabularnewline\n\\hline\n');



fprintf(fid, ' AlexNet (disCNN-iLab20M) ');
for i=1:4    
    fprintf(fid, ' & \\textbf{%.2f} ', 100*acc_deCNN_top5(i));
end
fprintf(fid, '\\tabularnewline\n\\hline\n');

fclose(fid);
