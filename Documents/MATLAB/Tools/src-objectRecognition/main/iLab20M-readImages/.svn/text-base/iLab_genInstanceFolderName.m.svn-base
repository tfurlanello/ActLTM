function name = iLab_genInstanceFolderName(args)
    
    opts = vl_argparse(iLab_validateImgFilePara, args); 
    suffix = '0000';    
    n = numel(num2str(opts.instance));
    suffix(end-n+1:end) = num2str(opts.instance);    
    name = [opts.class '-' suffix];

end