function name = iLab_idx2nameFocus(idx)

    if idx > 2 || idx < 0
        error('idx should be in range [0, 2]\n');
    end
    
    name = ['f' num2str(idx)];
   

end