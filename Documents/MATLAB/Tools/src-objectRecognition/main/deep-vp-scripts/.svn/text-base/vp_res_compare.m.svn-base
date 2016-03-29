
resSaveDir = '/lab/igpu3/u/kai/deep_vp/results/iros2016';

oldresFile = 'oldres.txt';

newresFile = 'deepVP-deviations.mat';



[routes_old, acc_old] = textread(fullfile(resSaveDir, oldresFile), '%s %f\n');

load(fullfile(resSaveDir, newresFile));

routes_new = u_routes;
acc_new = meDeviations_diffRoutes(:,1);


acc_new_align = zeros(1, numel(acc_old));
acc = zeros(numel(acc_old),2);
for i=1:numel(acc_old)
    
    i_old = acc_old(i);
    i_new = acc_new( strcmp(routes_new, routes_old{i}));
    
    acc(i,:) = [i_old i_new];
    
    
end


acc_new_align = acc(:,2);
[~, idx] = sort(acc_new_align, 'ascend');
acc = acc(idx,:);
routes = routes_old(idx);
nroutes = numel(routes);

routes_show  = routes;
for i=1:nroutes
    idx = strfind(routes{i}, '_');
    tmp = routes{i};
    tmp(idx) = '-';
    routes_show{i} = tmp;
    
end


figure;
plot(acc, 'linewidth', 3);

set(gca, 'ytick', 0:10:80, 'xtick', 1:nroutes, 'xticklabel', routes_show, ...      
    'xgrid', 'on', 'ygrid', 'on', 'fontsize', 20);

ylabel('mean L2 distance (pixels)', 'fontsize', 25);
legend({'Chang et al', 'deep VP'});



figure; 
bar(acc);
set(gca, 'ytick', 0:10:80, 'xtick', 1:nroutes, 'xticklabel', routes_show, ...      
    'xgrid', 'on', 'ygrid', 'on', 'fontsize', 20);

ylabel('mean L2 distance (pixels)', 'fontsize', 25);
legend({'Chang et al', 'deep VP'});


% append mAP to the end
acc = [acc; mean(acc)];
routes_show= cat(1, routes_show, 'total');
figure; 
bar(acc);
set(gca, 'ytick', 0:10:80, 'xtick', 1:(nroutes+1), 'xticklabel', routes_show, ...      
    'xgrid', 'on', 'ygrid', 'on', 'fontsize', 20);

ylabel('mean L2 distance (pixels)', 'fontsize', 25);
legend({'Chang et al', 'deep VP'});




