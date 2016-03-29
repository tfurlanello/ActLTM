

resSaveDir = '/lab/jiaping/papers/ECCV2016/results';

eval_deCNN = '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f18/iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000/maxActivation/test-evalInfo.mat';
eval_alexNet = '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f18/iLab20M-iLab_arc_de_dagnn_2streams_alexnet/maxActivation/test-evalInfo.mat';


% deCNN

imdb_eval   =  load(eval_deCNN);
evals       =  imdb_eval.eval;
clear imdb_eval;

gt      =   evals.gt(1,:);  
pred    =   squeeze(evals.pred(1,:,1));

gt      =   gt(:);
pred    =   pred(:);


% drawing
objects  = {'car', 'f1car', 'helicopter', 'plane', 'pickup', ...
                              'military', 'monster', 'semi', 'tank', 'van'};
figure;

plotCM(gt, pred, objects);
rotateXLabels(gca, 315 );% rotate the x tick
axis equal;
axis tight;
set(gcf, 'position', [200 200 1000 1000]);

% export_fig(fullfile(resSaveDir, 'CM-iLab20M-deCNN.pdf') , '-pdf', '-transparent', '-m1', gcf );


% alexnet

imdb_eval   =  load(eval_alexNet);
evals       =  imdb_eval.eval;
clear imdb_eval;

gt      =   evals.gt(1,:);  
pred    =   squeeze(evals.pred(1,:,1));

gt      =   gt(:);
pred    =   pred(:);


% drawing
objects  = {'car', 'f1car', 'helicopter', 'plane', 'pickup', ...
                              'military', 'monster', 'semi', 'tank', 'van'};
figure;

plotCM(gt, pred, objects);
rotateXLabels(gca, 315 );% rotate the x tick
axis equal;
axis tight;
set(gcf, 'position', [200 200 1000 1000]);

% export_fig(fullfile(resSaveDir, 'CM-iLab20M-alexNet.pdf') , '-pdf', '-transparent', '-m1', gcf );

