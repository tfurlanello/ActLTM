% to visualize receptive fields of a trained dagnn model

%% note: we can only run dagnn model

 
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                       image database file
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
imdb_file = '/lab/ilab/30/kai-vp/google_dataset-grey/vp-alexnet-dagnn-obj/imdb.mat';
imdb_file = '/home2/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2/imdb.mat';
imdb = load(imdb_file);

saveDir = '/lab/ilab/30/kai-vp/google_dataset-grey/visalization';
saveDir = '/home2/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2/visualization';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end


%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%%                              a classic linear-chain dagnn model
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
net_linear_file = '/lab/ilab/30/kai-vp/google_dataset-grey/vp-alexnet-dagnn-obj/net-epoch-20.mat';
net_linear_file = '/home2/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2/net-epoch-17.mat';
net_linear = load(net_linear_file);
net_linear = net_linear.net;
net_linear = dagnn.DagNN.loadobj(net_linear);
net_linear.conserveMemory = 0;
subsetTrain = find(imdb.images.set == 1);
subsetTest  = find(imdb.images.set == 3);
% sparseIdx   = 1:2:numel(subsetTest);
% subsetTest  = subsetTest(sparseIdx);
% subset      = [subsetTrain(:); subsetTest(:)];
subset = subsetTest;
locationID = vp_generate_Location_ID(imdb.images.name);
testID      = locationID(subset);
uTestID     = unique(testID);
nuTestID    = numel(uTestID);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- first save intermediate evaluation values 
modeltype           =   'dagnn';
whoseLabels         =   {'object', 'environment'};
whichLayersToEval   =   {'pool1out', 'pool2out', 'relu3out', 'relu4out', 'pool5out'};
%whichLayersToEval   =   {'dropout6out', 'dropout7out'};
evalFileName = 'imdb-simple-eval.mat';                                                

if ~exist(fullfile(saveDir, evalFileName), 'file')
    [~, imdb_simple] = iLab_cnn_predictBatch(net_linear, imdb, subset, modeltype,...
                                                        whoseLabels, whichLayersToEval);

    save(fullfile(saveDir, evalFileName), '-struct', 'imdb_simple', '-v7.3');
else
    imdb_simple = load(fullfile(saveDir, evalFileName));
end
simplennEval = imdb_simple.eval;
clear imdb_simple;

% return;

simplenn_saveDir = strcat(saveDir, filesep, 'visualization-me');
if ~exist(simplenn_saveDir, 'dir')
    mkdir(simplenn_saveDir);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- get receptive fields of designated layers
nStimuli = 100;
nIntermediateLayers = numel(whichLayersToEval);
assert( numel(simplennEval.intermediateLayers) == nIntermediateLayers);

    
    
    for l=1:nIntermediateLayers
        layerName   =   simplennEval.intermediateLayers(l).name;
        values      =   simplennEval.intermediateLayers(l).value;
        subset      =   simplennEval.subset;
        nchannels   =   size(values,1);

        sub_complex_saveDir = fullfile(rf_saveDir, layerName);
        if ~exist(sub_complex_saveDir, 'dir')
            mkdir(sub_complex_saveDir);
        end
        
        if ~exist(fullfile(sub_complex_saveDir, [layerName '.mat']), 'file')
            % entropy
            prob_obj = zeros(numel(idxObj), nchannels);
            prob_env = zeros(numel(idxEnv), nchannels);
            entropy_obj = zeros(1, nchannels);
            f_entropy_obj = zeros(1, nchannels) > 1.0;
            entropy_env = zeros(1, nchannels);
            f_entropy_env = zeros(1, nchannels) > 1.0;

            for c=1:nchannels            
                c_values = values(c,:);

                % obj
                for o=1:numel(idxObj)
                   tmp = gt_obj == idxObj(o);
                   prob_obj(o,c) = sum(c_values(tmp))/sum(tmp);
                end

                tmp = prob_obj(:,c);
                tmp = tmp/sum(tmp);
                tmpidx = (tmp ~= 0);
                tmp = tmp(tmpidx);

                if min(tmp) > 0
                    entropy_obj(c) =  -sum(tmp.*log2(tmp));
                    f_entropy_obj(c)  = 1;
                end

                % env
                for e=1:numel(idxEnv)
                   tmp = gt_env == idxEnv(e);
                   prob_env(e,c) = sum(c_values(tmp))/sum(tmp);                    
                end

                tmp = prob_env(:,c);
                tmp = tmp/sum(tmp);
                tmpidx = (tmp ~= 0);
                tmp = tmp(tmpidx);

                if min(tmp) > 0
                    entropy_env(c) =  -sum(tmp.*log2(tmp));
                    f_entropy_env(c) = 1;
                end
            end       

            save(fullfile(sub_complex_saveDir, [layerName '.mat']), ...
                        'prob_obj', 'prob_env', 'entropy_env', 'f_entropy_env', ...
                            'entropy_obj', 'f_entropy_obj');
        end
        
%         continue;
        
%         if l > nIntermediateLayers -2
%             continue;
%         end
        
        meCropCollage = cell(1, nchannels);
        % receptive fields
        for c=1:nchannels

            fprintf(1, 'processing: layer-%s, channel-%d\n', layerName, c);
            
            saveName = strcat(layerName, '-c', num2str(c));
            if exist(fullfile(sub_complex_saveDir, [saveName '-im.png']), 'file')
                continue;
            end

            c_values = values(c,:);
             
            [~, ind] = sort(c_values, 'descend');
            c_subset = subset(ind(1:nStimuli));
            whichLayerWhichChannel = struct('layer', layerName, ...
                                            'channel', c); 

            imdb_rf = iLab_dagnn_getImgCropFromReceptiveField(net, imdb, c_subset, ...
                                                        whichLayerWhichChannel, whoseLabels);
%             save(fullfile(sub_complex_saveDir, [saveName '.mat']), '-struct', 'imdb_rf');

            rawImgs = [];
            crops = [];

            for s=1:numel(imdb_rf.stimuli.crop)
                tmpImg = imdb_rf.stimuli.crop(s).image;
                rawImgs = cat(2, rawImgs, tmpImg(:));
                tmpPatch = imdb_rf.stimuli.crop(s).patch;
                crops = cat(2, crops, tmpPatch(:));
            end
            
            szImg = round(sqrt(size(rawImgs,1)/3));
            szCrop = round(sqrt(size(crops,1)/3)); 
            % save mean images
            me_rawImg = uint8(mean(rawImgs,2));
            me_cropImg = uint8(mean(crops,2));
            me_rawImg = reshape(me_rawImg, [szImg szImg 3]);
            me_cropImg = reshape(me_cropImg, [szCrop szCrop 3]);

            imwrite(me_rawImg, fullfile(sub_complex_saveDir, [saveName '-im-me.png']), 'png');
            imwrite(me_cropImg,   fullfile(sub_complex_saveDir, [saveName '-rf-me.png']), 'png');
            
            meCropCollage{c} = me_cropImg(:);
            % save receptive fields
            rawImgs = uint8(rawImgs);
            crops = uint8(crops);

            rawImgs = rawImgs(:,1:nShow);
            crops = crops(:,1:nShow);
            
            rawImgs = imCollage(rawImgs, [szImg szImg]);
            crops = imCollage(crops, [szCrop szCrop]);

            imwrite(rawImgs, fullfile(sub_complex_saveDir, [saveName '-im.png']), 'png');
            imwrite(crops,   fullfile(sub_complex_saveDir, [saveName '-rf.png']), 'png');

        end
        if ~exist(fullfile(sub_complex_saveDir, [layerName '-collage.png']), 'file')
            meCropCollageImg = imCollage(cell2mat(meCropCollage), [szCrop szCrop]);
            imwrite(meCropCollageImg, fullfile(sub_complex_saveDir, [layerName '-collage.png']), 'png');
            save(fullfile(sub_complex_saveDir, [layerName '-collage.mat']), 'meCropCollage');
        end
        
        
    end
     
    % t-SNE dimension reduction
% 	fc2values      =   complexEval.intermediateLayers(nIntermediateLayers-3).value;
%     labels = complexEval.gt;          
%     [fc22D, landmarks] = visualizeDescriptorsTSNE(fc2values', labels);            
%     fc22D = fc22D(landmarks,:);
% %     complexEval =   rmfield(complexEval, {'intermediateLayers'}) ;
%     complexEvalfc.fc22D  =   fc22D;
%     complexEvalfc.fc2    =   fc2values;
%     complexEvalfc.gt     = complexEval.gt;
%     complexEvalfc.pred   = complexEval.pred;
%     complexEvalfc.subset = complexEval.subset;
%     
%     save(fullfile(sub_complex_saveDir, 'tSNE-fc2-2D.mat'), '-struct', 'complexEvalfc' , '-v7.3');
%     clear complexEvalfc;
%     close all;
    



 

clear all;