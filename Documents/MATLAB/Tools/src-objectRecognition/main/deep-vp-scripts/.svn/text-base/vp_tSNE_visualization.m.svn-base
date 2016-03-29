load('/lab/jiaping/igpu3home2/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2/visualization/tSNE-fc2-2D.mat');
saveDir = '/lab/jiaping/igpu3home2/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2/visualization';


% fix x, draw y
% gty = pred(1,:);
% gtx = pred(2,:);
% 
% 
% ugtx = unique(gtx);
% ugty = unique(gty);
% 
% label = gty;
% 
% ulabel = unique(label);
% figure;
% for x=1:numel(ugtx)
%     
%     figure;
%     idx = gtx == ugtx(x);
%     fcx = fc22D(idx,:);
%     labelx = label(idx);
% for l=1:numel(ulabel)
%     idx = labelx == ulabel(l);
%     
%     fc = fcx(idx,:);
%     c = [rand rand rand];
%     scatter(fc(:,1), fc(:,2), 'markeredgecolor', c, 'markerfacecolor', c); hold on;
%     
% end
% 
% end
%  
% axis equal; axis tight;
% set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1], 'xticklabel', [], 'yticklabel', []);
% set(gcf, 'Units', 'normalized', 'Position', [0,0,1,1] );
% export_fig(fullfile(saveDir, 'cluster225.png') , '-png', '-m2', gcf );
% export_fig(fullfile(saveDir, 'cluster225.pdf') , '-pdf', '-transparent', '-m1', gcf );





% draw clusters
gty = pred(1,:);
gtx = pred(2,:);

label = (gtx - 1)*15 + gty;

ulabel = unique(label);
figure;
for l=1:numel(ulabel)
    idx = label == ulabel(l);
    
    fc = fc22D(idx,:);
    c = [rand rand rand];
    scatter(fc(:,1), fc(:,2), 'markeredgecolor', c, 'markerfacecolor', c); hold on;
    
end
 
axis equal; axis tight;
set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1], 'xticklabel', [], 'yticklabel', []);
set(gcf, 'Units', 'normalized', 'Position', [0,0,1,1] );
export_fig(fullfile(saveDir, 'cluster225.png') , '-png', '-m2', gcf );
export_fig(fullfile(saveDir, 'cluster225.pdf') , '-pdf', '-transparent', '-m1', gcf );



% draw heading
gtx = pred(2,:);

ugtx = unique(gtx);
figure;
for l=1:numel(ugtx)
    idx = gtx == ugtx(l);
    
    fc = fc22D(idx,:);
    c = [rand rand rand];
    scatter(fc(:,1), fc(:,2), 'markeredgecolor', c, 'markerfacecolor', c); hold on;
    
end

title('heading', 'fontsize', 40);
set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1], 'xticklabel', [], 'yticklabel', []);
axis equal; axis tight;
set(gcf, 'Units', 'normalized', 'Position', [0,0,1,1] );
export_fig(fullfile(saveDir, 'cluster-heading.png') , '-png', '-m2', gcf );
export_fig(fullfile(saveDir, 'cluster-heading.pdf') , '-pdf', '-transparent', '-m1', gcf );

% draw pitch
gty = pred(1,:);

ugty = unique(gty);
figure;
for l=1:numel(ugty)
    idx = gty == ugty(l);
    
    fc = fc22D(idx,:);
    c = [rand rand rand];
    scatter(fc(:,1), fc(:,2), 'markeredgecolor', c, 'markerfacecolor', c); hold on;
    
end

title('pitch', 'fontsize', 40);
axis equal; axis tight;
set(gca, 'xcolor', [1 1 1], 'ycolor', [1 1 1], 'xticklabel', [], 'yticklabel', []);
set(gcf, 'Units', 'normalized', 'Position', [0,0,1,1] );
export_fig(fullfile(saveDir, 'cluster-pitch.png') , '-png', '-m2', gcf );
export_fig(fullfile(saveDir, 'cluster-pitch.pdf') , '-pdf', '-transparent', '-m1', gcf );

close all;
