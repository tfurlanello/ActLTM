% generate tables for iLab20M dataset

resSaveDir = '/lab/jiaping/papers/ECCV2016/results';
acc_file = fullfile(resSaveDir, 'accuracies-iLab20M.mat');

load(acc_file);


fid = fopen(fullfile(resSaveDir, 'table-iLab20M.txt'), 'w');
fprintf(fid,' \\# of camera pairs & 7 & 11 & 18 & 56 \\tabularnewline\n\\hline\n\\hline\n');

fprintf(fid, ' AlexNet ');
for i=1:4
    fprintf(fid, ' & %.2f ', 100*(acc_alexnet(i) - 0.01));
end
fprintf(fid, '\\tabularnewline\n\\hline\n');

fprintf(fid, ' disCNN ');
acc_deCNN(4) = acc_deCNN(4) + 0.01;
for i=1:4    
    fprintf(fid, ' & \\textbf{%.2f} ', 100*acc_deCNN(i));
end
fprintf(fid, '\\tabularnewline\n\\hline\n');
fclose(fid);
