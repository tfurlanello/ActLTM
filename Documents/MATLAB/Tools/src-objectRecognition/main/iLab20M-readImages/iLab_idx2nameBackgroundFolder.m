function name = iLab_idx2nameBackgroundFolder(idx)

    
    suffix = '0000';
    
    n = numel(num2str(idx));
    suffix(end-n+1:end) = num2str(idx);
    
    name = ['background-' suffix];
   

end