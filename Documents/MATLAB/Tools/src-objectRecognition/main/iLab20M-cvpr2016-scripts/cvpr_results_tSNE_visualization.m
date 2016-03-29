% cvpr tSNE visualization

maxsumFolderNames =  {'maxActivation', 'sumActivation'};
evalFileName      =  'imdb-eval.mat';
nActivationTypes = numel(maxsumFolderNames);
%% (1) compute correlation of pose and identity entropy

alexDir = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-obj/visualization';
w2cnn_I = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-fc2/visualization'; 
w2cnn_MI = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/visualization'; 

% imdb = load('/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/imdb.mat');

%
gap      =  10;
sidx     =  5;
nObjects =  10;
nPose    =  88;
makers   =  {'o','s','^','p', '*'};

colors = zeros(3, nObjects);
makertypes = cell(1,nObjects);
for c=1:nObjects
    colors(:,c) = [rand rand rand];
    makertypes{c} = makers{rem(c,5)+1};
end

%% (1) compute viewpoint-wise accuracies
% acc_viewpoint = cell(nActivationTypes, 2);
% for a=1:nActivationTypes
%     
%     load(fullfile(w2cnn_MI, maxsumFolderNames{a}, 'tSNE-fc2-2D-correct2.mat'));
%     
%     gt_pose = gt(2,:);
%     gt_obj = gt(1,:);
%     pred_obj = pred(1,:);
%     acc = zeros(nPose,1);
%     for p=1:nPose
%         tmpidx = gt_pose == p;
%         acc(p) = sum(gt_obj(tmpidx) == pred_obj(tmpidx))/sum(tmpidx);
%     end
%     acc_viewpoint{a,1} = acc;
%     
%     load(fullfile(alexDir, maxsumFolderNames{a}, 'tSNE-fc2-2D-correct2.mat') );
%     gt_obj = gt(1,:);
%     pred_obj = pred(1,:);
%     acc = zeros(nPose,1);
%     for p=1:nPose
%         tmpidx = gt_pose == p;
%         acc(p) = sum(gt_obj(tmpidx) == pred_obj(tmpidx))/sum(tmpidx);
%     end
%     acc_viewpoint{a,2} = acc;
%     
% end
% 
% bigidx = find( acc_viewpoint{1,1} > acc_viewpoint{1,2} + 0.10);
% selectiveIdx = [];
% 
% for i=1:numel(bigidx)    
%     selectiveIdx = cat(2, selectiveIdx, find(bigidx(i) == gt_pose));
% end


%% another way to select which points to show
% idx_diff = cell(1,nActivationTypes);
% for a=1:nActivationTypes
%     
%     load(fullfile(w2cnn_MI, maxsumFolderNames{a}, 'tSNE-fc2-2D-correct2.mat'));
%     
%     gt_pose = gt(2,:);
%     gt_obj = gt(1,:);
%     pred_obj = pred(1,:);
% 
%     idx_ours_wrong = find(gt_obj ~= pred_obj);
%     
%     load(fullfile(alexDir, maxsumFolderNames{a}, 'tSNE-fc2-2D-correct2.mat') );
%     gt_obj = gt(1,:);
%     pred_obj = pred(1,:);
% 
%     idx_alex_wrong = find(gt_obj ~= pred_obj);    
%     idx_diff{a} = setdiff(idx_ours_wrong, idx_alex_wrong);
%     
% end
% finalIdx = [setdiff(idx_diff{1}, selectiveIdx) idx_diff{1}];

%% (2) visualization

for a=1:nActivationTypes
    
    load(fullfile(w2cnn_MI, maxsumFolderNames{a}, 'tSNE-fc2-2D-correct5.mat'));
  
    nSamples =  size(fc22D,1);
%     idx      =  sidx:gap:nSamples;   
    idx = 1:nSamples;
    feat     =  fc22D(idx,:);
    labels   =  gt(1,idx);
    i_subset =  subset(idx);
    
%     imageFiles = strcat([imageDir filesep], images.name(i_subset));
%     [nonsquare, square] = tSNEcollage(feat, imageFiles);

    
    figure;
    for o=1:nObjects
        tmp = labels == o;
        feato = feat(tmp,:);
        scatter(feato(:,1), feato(:,2), 80, makertypes{o}, 'MarkerEdgeColor', colors(:,o), ...
                        'MarkerFaceColor', colors(:,o),  'linewidth', 5); hold on;        
        
    end
%     title('w2cnn');
        axis square;
        axis tight;
        set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1], 'xticklabel', [], 'yticklabel', []);
        set (gcf, 'Units', 'normalized', 'Position', [0,0,1,1]);
%     export_fig(['fig-tsen-2wcnn' maxsumFolderNames{a} '.pdf'], '-pdf', '-m1', '-transparent', gcf);
    
    load(fullfile(alexDir, maxsumFolderNames{a}, 'tSNE-fc2-2D-correct5.mat') );
    nSamples = size(fc22D,1);
%     idx = sidx:gap:nSamples;    
    idx = 1:nSamples;
    feat = fc22D(idx,:);
    labels = gt(1,idx);
    
    figure;
    for o=1:nObjects
        tmp = labels == o;
        feato = feat(tmp,:);
        scatter(feato(:,1), feato(:,2), 80, makertypes{o}, 'MarkerEdgeColor', colors(:,o), ...
            'MarkerFaceColor', colors(:,o),  'linewidth', 5); hold on;        
        
    end
%     legend(imdb.classes.name{1}, 'fontsize', 40);
    
%     title('alexnet');
    axis square;
    axis tight;
    set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1], 'xticklabel', [], 'yticklabel', []);
    set (gcf, 'Units', 'normalized', 'Position', [0,0,1,1]);
%     export_fig(['fig-tsen-alexnet' maxsumFolderNames{a} '.pdf'], '-pdf', '-m1', '-transparent', gcf);
       
end


