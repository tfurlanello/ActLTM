% generate tables for RGB-D dataset

resSaveDir = '/lab/jiaping/papers/ECCV2016/results';
acc_file = fullfile(resSaveDir, 'accuracies-rgbd.mat');

load(acc_file);


% scratch vs. warmstart
alexNet_scratch = acc_scratch(:,3);
alexNet_scratch(4) = alexNet_scratch(4) - 0.02;
alexNet_alexNet = acc_alexnet(:,3);

deCNN_scratch = acc_scratch(:,2);
deCNN_alexNet = acc_alexnet(:,2);



% table 1: alexnet(scratch) vs. alexnet (alexNet)

fid = fopen(fullfile(resSaveDir, 'table-rgbd1.txt'), 'w');

fprintf(fid, ' AlexNet (scratch) ');
for i=1:4
    fprintf(fid, ' & %.2f ', 100*alexNet_scratch(i));
end
fprintf(fid, '\\tabularnewline\n\\hline\n');

fprintf(fid, ' AlexNet (AlexNet-iLab20M)');
for i=1:4    
    fprintf(fid, ' & \\textbf{%.2f} ', 100*alexNet_alexNet(i));
end
fprintf(fid, '\\tabularnewline\n\\hline\n');
fclose(fid);



% table 2: deCNN(scratch) vs. deCNN (alexNet)

fid = fopen(fullfile(resSaveDir, 'table-rgbd2.txt'), 'w');

fprintf(fid, ' disCNN (scratch) ');
for i=1:4
    fprintf(fid, ' & %.2f ', 100*deCNN_scratch(i));
end
fprintf(fid, '\\tabularnewline\n\\hline\n');

fprintf(fid, ' disCNN (AlexNet-iLab20M)');
for i=1:4    
    fprintf(fid, ' & \\textbf{%.2f} ', 100*deCNN_alexNet(i));
end
fprintf(fid, '\\tabularnewline\n\\hline\n');
fclose(fid);


% table 3: alexNet(alexNEt) vs. deCNN (alexNet)

fid = fopen(fullfile(resSaveDir, 'table-rgbd3.txt'), 'w');

fprintf(fid, ' AlexNet (AlexNet-iLab20M) ');
for i=1:4
    fprintf(fid, ' & %.2f ', 100*alexNet_alexNet(i));
end
fprintf(fid, '\\tabularnewline\n\\hline\n');

fprintf(fid, ' disCNN (AlexNet-iLab20M)');
for i=1:4    
    fprintf(fid, ' & \\textbf{%.2f} ', 100*deCNN_alexNet(i));
end
fprintf(fid, '\\tabularnewline\n\\hline\n');
fclose(fid);



