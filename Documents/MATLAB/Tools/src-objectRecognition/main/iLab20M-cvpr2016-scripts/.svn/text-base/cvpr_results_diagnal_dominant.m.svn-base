

alexDir = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-obj/visualization';
w2cnn_I = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-fc2/visualization'; 
w2cnn_MI = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/visualization'; 


distsaveName = 'dotproduct.mat';

%fc7
% alex net

load(fullfile(alexDir, 'maxActivation', distsaveName));
obj2obj_dist_alexnet = obj2obj_dist;



dist = obj2obj_dist_alexnet{7};

dist = dist + dist' - diag(diag(dist));

diag_donimant_alexnet7 = diag(dist)./ ...
                    (sum(dist,2) - diag(dist));


% ours
load(fullfile(w2cnn_MI, 'maxActivation', distsaveName));
obj2obj_dist_ours = obj2obj_dist;


dist = obj2obj_dist_ours{7};

dist = dist + dist' - diag(diag(dist));

diag_donimant_ours7 = diag(dist)./ ...
                    (sum(dist,2) - diag(dist));
                
                
 show7 = [diag_donimant_alexnet7 diag_donimant_ours7]
                
%%  fc6
dist = obj2obj_dist_alexnet{6};

dist = dist + dist' - diag(diag(dist));

diag_donimant_alexnet6 = diag(dist)./ ...
                    (sum(dist,2) - diag(dist));


% ours
 


dist = obj2obj_dist_ours{6};

dist = dist + dist' - diag(diag(dist));

diag_donimant_ours6 = diag(dist)./ ...
                    (sum(dist,2) - diag(dist));


show6 = [diag_donimant_alexnet6 diag_donimant_ours6]


% drawing
objects  = {'car', 'f1car', 'helicopter', 'plane', 'pickup', ...
                              'military', 'monster', 'semi', 'tank', 'van'};
figure;
dist = obj2obj_dist_alexnet{6};
dist = dist + dist' - diag(diag(dist));

draw_cm(dist, objects,10);
rotateXLabels(gca, 315 );% rotate the x tick

export_fig('alexnet-diagnal-metric.pdf' , '-pdf', '-transparent', '-m1', gcf );

figure;

dist = obj2obj_dist_ours{6};
dist = dist + dist' - diag(diag(dist));
draw_cm(dist, objects,10);
rotateXLabels(gca, 315 );% rotate the x tick
export_fig('ours-diagnal-metric.pdf' , '-pdf', '-transparent', '-m1', gcf );



