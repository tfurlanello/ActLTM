function name = iLab_idx2nameCamera(idx)
    
    if idx > 10 || idx < 0
        error('idx should be in range [0, 10]\n');
    end
    
    suffix = '00';
    
    n = numel(num2str(idx));
    suffix(end-n+1:end) = num2str(idx);
    
    name = ['c' suffix];
    
end