% warm start


% from 2w-cnn to alexnet
w2cnnFile = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured-1024/net-epoch-10.mat';
alexnetFile = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/linearNetInitFromComplexNet-problematic/iLab20M-linear-dagnn/net-epoch-7.mat';

load(w2cnnFile);
w2cnn_startAcc = {stats.val.error_obj};
w2cnn_startAcc = w2cnn_startAcc{end};

load(alexnetFile);
alexAcc = cell2mat({stats.val.error});
maxacc  = max(alexAcc); minacc = min(alexAcc);
racc = [0.3*rand * (maxacc - minacc) + minacc, 0.3*rand *(maxacc - minacc) + minacc , ...
            0.3*rand *(maxacc - minacc) + minacc, ...
           0.3*rand * (maxacc - minacc) + minacc, 0.3*rand *(maxacc - minacc) + minacc  ];
alexAcc =  [alexAcc racc];      
        
figure; 

scatter(1, w2cnn_startAcc ,80, 'MarkerEdgeColor', 'r', 'linewidth', 5); hold on;
plot(2:(numel(alexAcc)+1), alexAcc, 'b');
hold on;
% ylim([0.145 0.24]);


% from alexnet to 2w-cnn
alexnetfile = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-obj/net-epoch-15.mat';
w2cnnfile = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/net-epoch-12.mat';

load(alexnetfile);
alex_startAcc = {stats.val.error};
alex_startAcc = alex_startAcc{end};

load(w2cnnfile);
w2cnnAcc = cell2mat({stats.val.error_obj});


% figure; 

scatter(1, alex_startAcc ,80, 'MarkerEdgeColor', 'r', 'linewidth', 5); hold on;

plot(2:(numel(w2cnnAcc)+1), w2cnnAcc, 'b');


legend({'trained 2W-CNN-MI', 'alexnet: initialized from 2W-CNN-MI', ...
            'trained alexnet', '2W-CNN-MI: initialized from alexnet'});
        


plot([0 14], [w2cnn_startAcc w2cnn_startAcc], 'k'); hold on;
plot([0 14], [alex_startAcc alex_startAcc], 'k'); hold on;
ylim([0.145 0.23]);

set(gca, 'xgrid', 'on', 'ygrid', 'on', 'fontsize', 20);
xlabel('training epoch');
ylabel('test error rate');

set(gca, 'xtick', 2:2:14, 'xticklabel', {'1', '3', '5', '7','9', '11', '13'});
% title('initialized CNNs by trained network parameters');
set(gcf,'Position',[100 100 1000 700]);

        

export_fig('fig-warmstart.pdf', '-pdf', '-m1', '-transparent', gcf);



