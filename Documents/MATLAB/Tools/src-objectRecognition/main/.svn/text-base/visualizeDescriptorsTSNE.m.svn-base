function [mapped2D, landmarks] = visualizeDescriptorsTSNE(descriptors, labels)

%FAST_TSNE Runs the fast Intel (IPP) implementation of t-SNE
%
%   [mappedX, landmarks, costs] = fast_tsne(X, no_dims, initial_dims, landmarks, perplexity)
%
% Runs the fast implementation Diffusion Stochastic Neighbor Embedding 
% algorithm. The high-dimensional datapoints are specified by X. First, the
% dimensionality of the datapoints is reduced to initial_dims dimensions
% using PCA (default = 30). Then, Diffusion SNE reduces the points to 
% no_dims dimensions (default = 2). The percentage of points to use as 
% landmarks may be specified through landmarks (0 <= landmarks <= 1), or 
% you may specify the indices of the landmark points in a vector. If not 
% specified, both the number and the amount of landmarks points is 
% determined automatically.  The used perplexity in the Gaussian kernel can 
% be set through the perplexity variable (default = 30). Note that the 
% perplexity is mainly of influence on small datasets where landmarks are 
% not employed.
% The function returns the low-dimensional datapoints in mappedX, the used 
% landmark points in landmarks, and the cost contribution per datapoint in
% costs.
%
%
% (C) Laurens van der Maaten
% Maastricht University, 2008

    addpath(genpath('/lab/jiaping/projects/google-glass-project/src/t-SNE'));
    [mappedX, landmarks, costs] = fast_tsne(descriptors, [], [], 1); %, no_dims, initial_dims, landmarks, perplexity);
    
%     if ismatrix(sequences)
%         sequences = mat2cell(sequences, ones(1, size(sequences,1)));
%     end
    labels      =   labels(landmarks);
    mapped2D    =   mappedX;
    
    return;    
    x = mappedX;
    x = bsxfun(@minus, x, min(x));
    x = bsxfun(@rdivide, x, max(x));
    mappedX = x;
    
    markerShapes = {'o', 's', 'd', 'p'};
    n = ceil(length(unique(labels))/4);
    shapes = {};
    for i=1:n
        shapes = [shapes markerShapes];
    end
    shapes = shapes(1:length(unique(labels)));
	colors = rand(length(unique(labels)),3);
    property2labels = unique(labels);
    property2labels = sort(property2labels, 'ascend');
    colorsPts = zeros(size(mappedX,1),3);
    shapePts = cell(size(mappedX,1),1);
    for i=1:size(mappedX,1)
        colorsPts(i,:) = colors(property2labels == labels(i),:); 
        shapePts{i} = shapes{property2labels == labels(i)};
    end
    
    % in order to draw legend
    uni_labels = sort(unique(labels), 'ascend');
    legend_sequences    =   cell(length(uni_labels),1);
    legend_mappedX      =   zeros(length(uni_labels),2);
    legend_colors       =   zeros(length(uni_labels),3);
    legend_shapes       =   cell(length(uni_labels),1);
    for i=1:length(uni_labels)
        imappedX = mappedX(labels == uni_labels(i),:);
        legend_mappedX(i,:)  = imappedX(1,:);
        legend_colors(i,:) = colors(property2labels == uni_labels(i),:);
        legend_shapes{i} = shapes{property2labels == uni_labels(i)};
    end
   colorsPts    = [legend_colors; colorsPts];
   shapePts     = [legend_shapes; shapePts];
   mappedX      = [legend_mappedX; mappedX];

    hdot = figure;
    set (hdot,'Units', 'normalized', 'Position', [0,0,1,1]);
    axes('Position', [0 0 1 1]); 
    for i=1:size(mappedX,1) 
        plot(mappedX(i,1), mappedX(i,2), 'Marker', shapePts{i}, 'MarkerSize', 8, ...
                'MarkerFaceColor', colorsPts(i,:), 'MarkerEdgeColor', colorsPts(i,:)); hold on; 
    end
%     legend(activityNames, 'FontSize', 20);
    set(gca, 'XTick', [], 'YTick', []);   
    axis equal;
    
end