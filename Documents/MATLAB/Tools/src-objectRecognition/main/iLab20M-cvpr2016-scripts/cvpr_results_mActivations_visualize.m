% visualize mean activations
 
saveDir = '';

maxsumFolderNames =  {'maxActivation', 'sumActivation'};
evalFileName      =  'imdb-eval.mat';
nActivationTypes = numel(maxsumFolderNames);
%% (1) compute correlation of pose and identity entropy

alexDir = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016-e2/iLab20M-alexnet-dagnn-obj/visualization';
w2cnn_I = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-fc2/visualization'; 
w2cnn_MI = '/lab/jiaping/igpu3home2/u/jiaping/iLab20M-objRec/results/cvpr2016/iLab20M-alexnet-dagnn-multiLevelInjection-conv1234fc2/visualization'; 


whichLayersToEval   =  {'pool1out', 'pool2out', 'relu3out', 'relu4out', ...
                                 'pool5out', 'dropout6out', 'dropout7out'};
layerNames          =  {'pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7'};                                 
nIntermediateLayers =  numel(whichLayersToEval);                                 
nEvals              =  numel(whichLayersToEval);     


for a=nActivationTypes:-1:1
    
   w2cnn_MI_a = fullfile(w2cnn_MI, maxsumFolderNames{a}, 'mActivations.mat');
   alexnet_a  = fullfile(alexDir, maxsumFolderNames{a}, 'mActivations.mat');
   w2cnn_I_a  = fullfile(w2cnn_I, maxsumFolderNames{a}, 'mActivations.mat');
   
   load(w2cnn_MI_a);
   act_MI = mActivations;
   
   
   load(w2cnn_I_a);
   act_I = mActivations;
      
   
   load(alexnet_a);
   act_alex = mActivations;
%    
%    for l=1:nIntermediateLayers
%        figure; 
%        hist([act_I{l} act_alex{l}]);
%    end
   
   
end



for l=1:nIntermediateLayers
    % 2w-cnn-MI
    
    act = act_MI{l};    
    nUnits = numel(act); 
%     figure; hist(act);

    if l~=2
        [f_MI,xi_MI] = ksdensity(act);
    else
        [f_MI,xi_MI] = ksdensity(act, 'bandwidth', 8);
    end
 
    
    % alexnet
        act = act_alex{l};    
    nUnits = numel(act); 
%     figure; hist(act);
    [f_alex,xi_alex] = ksdensity(act);
    
    figure; plot(xi_MI, f_MI, 'linewidth', 5, 'color', 'b'); hold on;
    plot(xi_alex, f_alex, 'linewidth', 5, 'color', [0.87 0.49 0]);
    set(gca, 'fontsize', 15);
    ylabel('probability density', 'fontsize', 25);
    xlabel('mean activations', 'fontsize', 25);
    xlim([min([xi_alex xi_MI]) max([xi_alex xi_MI])]);
    ylim([min([f_alex f_MI]) max([f_alex f_MI])]);
    title(layerNames{l}, 'fontsize', 30);
    
    if l==1
        legend({'2W-CNN-MI', 'alexnet'});
    end
    
    set(gcf,'Position',[500 500  600 400]);
    
    export_fig(['fig-mResponse-', layerNames{l} '.pdf'] , '-pdf', '-m1', '-transparent',   gcf );

       
    
end

















