resSaveDir = '/lab/igpu3/u/kai/deep_vp/results/iros2016';

mapping_label2coord = '/lab/igpu3/u/kai/deep_vp/results/google_dataset/vp-alexnet-simplenn-obj/mapping.mat';
load(mapping_label2coord);

saveDirs = {...
       '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc1024', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc512', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc256', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc128', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc64', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc32', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc16'};

resFileName = 'prediction-prob.mat';
capacities = {'fc1024', 'fc512', 'fc256', 'fc128', 'fc64' , 'fc32', 'fc16'};
nCapacities = numel(capacities);

topHits = [1 2 3 4 5];
nTopHits = numel(topHits);
accuracies = zeros(nCapacities,nTopHits);


meDeviations = zeros(nCapacities,1);
meDeviations_diffRoutes = zeros(nCapacities, 24);

for i=1:nCapacities
    accFile = fullfile(saveDirs{i}, resFileName);
    load(accFile);
    
    % extract route
    
    if i==1
        routes = cell(numel(testFiles),1);
        idx = strfind(testFiles, '/');
        for t=1:numel(testFiles)
            routes{t} = testFiles{t}(idx{t}(7)+1:idx{t}(8)-1);
        end
        [u_routes, ~, idx] = unique(routes);
    end
    
    
    gt   = labels_gt;
    pred = labels_pred;
    
    nTest = numel(gt);
    
    deviations = zeros(nTest,1);
    for t=1:nTest
        refx = mapping(gt(t),2);
        refy = mapping(gt(t),3);
        
        t_pred = pred(t,:);
        t_pred_prob = probs(t,:);
        predxs = mapping(t_pred, 2);
        predys = mapping(t_pred, 3);
        
        predx = predxs(:)' * t_pred_prob(:);
        predy = predys(:)' * t_pred_prob(:);
        
        deviations(t) = sqrt( max((predx - refx)^2 + (predy - refy)^2,0));
        
    end
   
    meDeviations(i) = mean(deviations);
    
    for d=1:24
        meDeviations_diffRoutes(i,d) = mean(deviations(idx==d));
    end
    
    
end
meDeviations_diffRoutes = meDeviations_diffRoutes'; 

save(fullfile(resSaveDir, 'deepVP-deviations.mat'), 'meDeviations_diffRoutes', 'capacities', ...
                        'meDeviations', 'u_routes');
% save results
fid = fopen(fullfile(resSaveDir, 'deepVP-deviations.txt'), 'w');
fprintf(fid, 'datasets fc1024 fc512 fc256 fc128 fc64 fc32 fc16\n');
fprintf(fid, '         %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n', ...
        meDeviations(1), meDeviations(2), meDeviations(3), meDeviations(4), ...
        meDeviations(5), meDeviations(6), meDeviations(7));
    

for d=1:24
    fprintf(fid, '%s  %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n', ...
        u_routes{d},  meDeviations_diffRoutes(d,1), meDeviations_diffRoutes(d,2), meDeviations_diffRoutes(d,3), meDeviations_diffRoutes(d,4), ...
        meDeviations_diffRoutes(d,5), meDeviations_diffRoutes(d,6), meDeviations_diffRoutes(d,7));
end

fclose(fid);
    
% save table
fid = fopen(fullfile(resSaveDir, 'deepVP-deviations.table'), 'w');
fprintf(fid, 'datasets & fc1024 & fc512 & fc256 & fc128 & fc64 & fc32 & fc16 \\tabularnewline\n');
    
for d=1:24
    fprintf(fid, '%s & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f \\tabularnewline\n', ...
        u_routes{d},  meDeviations_diffRoutes(d,1), meDeviations_diffRoutes(d,2), meDeviations_diffRoutes(d,3), meDeviations_diffRoutes(d,4), ...
        meDeviations_diffRoutes(d,5), meDeviations_diffRoutes(d,6), meDeviations_diffRoutes(d,7));
end

fclose(fid);

% save figure

figure;   
% plot(meDeviations, 'linewidth', 3);
bar(meDeviations);

set(gca, 'ytick', 0:10:100,  'xtick', 1:nCapacities, 'xticklabel', capacities, ...      
    'xgrid', 'on', 'ygrid', 'on', 'fontsize', 20);
xlim([0 nCapacities+1]);
ylim([0 100]);
xlabel('# of nodes in fc layers', 'fontsize', 25);
ylabel('mean deviation from GT (pixels)', 'fontsize', 25); 
title('mean deviation vs. network capacities', 'fontsize', 30);
set(gcf, 'position', [500 500 1000 700]);

export_fig(fullfile(resSaveDir, 'deepVP-deviations.pdf'), '-pdf', '-m1', '-transparent', gcf);







