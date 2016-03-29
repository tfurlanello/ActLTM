function varargout = rgbd_genImgPairs(frames, cameraIdx, gap, repetition)
    
    assert(nargin >= 2);
    if ~isvector(frames) || ~isscalar(cameraIdx)
        error('A sequence of frames under one fixed camera are the expected inputs\n');
    end
    
    if ~exist('gap', 'var') || isempty(gap)
        gap = 5;
    end
    
    if ~exist('repetition', 'var') || isempty(repetition)
        repetition = 4;
    end
    
    frames = frames(:);
    
    [s_frames, ~]   =  sort(frames,'ascend'); 
    
    if numel(s_frames) < gap
        pairs_frame = [];
        pairs_label = [];
        map2frameIdx = [];
        varargout = {pairs_frame, pairs_label, map2frameIdx};
        return;
    end
    
    chosen_frames_l =  s_frames(randi(gap)):gap:s_frames(end);
    chosen_frames_l = chosen_frames_l(:);
    len_ref = numel(chosen_frames_l);
    
    pairs_frame = [];
    pairs_label = [];
    
    for r=1:repetition        
        chosen_frames_r = chosen_frames_l + r*gap;        
        pairs_frame = cat(1, pairs_frame, [chosen_frames_l chosen_frames_r]);        
        pairs_label = cat(1, pairs_label, repmat([cameraIdx r*gap],len_ref,1));        
    end
    
    % make sure each image-pair does exist!
    bexist = sum(ismember(pairs_frame, frames),2) == 2;
    pairs_frame = pairs_frame(bexist,:);
    pairs_label = pairs_label(bexist,:);
    nPairs = size(pairs_frame,1);
    % build mapping from pairs_frame to the original pairs
    map2frameIdx = zeros(nPairs,2);
    
    for f=1:nPairs
       map2frameIdx(f,1) = find(pairs_frame(f,1) == frames);
       map2frameIdx(f,2) = find(pairs_frame(f,2) == frames);
    end


    varargout = {pairs_frame, pairs_label, map2frameIdx};


    
end