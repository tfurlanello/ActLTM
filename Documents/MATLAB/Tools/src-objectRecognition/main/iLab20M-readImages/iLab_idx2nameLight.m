function name = iLab_idx2nameLight(idx)

    if idx > 4 || idx < 0
        error('idx should be in range [0, 4]\n');
    end
    
    name = ['l' num2str(idx)];
   

end