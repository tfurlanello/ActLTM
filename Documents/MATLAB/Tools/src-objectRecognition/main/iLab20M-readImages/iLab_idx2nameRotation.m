function name = iLab_idx2nameRotation(idx)


    if idx > 7 || idx < 0
        error('idx should be in range [0, 7]\n');
    end
    
    suffix = '00';
    
    n = numel(num2str(idx));
    suffix(end-n+1:end) = num2str(idx);
    
    name = ['r' suffix];
   

end