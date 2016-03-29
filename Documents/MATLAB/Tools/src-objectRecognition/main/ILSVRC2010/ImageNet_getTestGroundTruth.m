function labels =  ImageNet_getTestGroundTruth
    
    gtfile = '/lab/igpu3/u/jiaping/imageNet2010/images/test-ground-truth.txt'; 
    labels = textread(gtfile, '%s\n');    
    labels = str2double(labels);
    

end