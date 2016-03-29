function synsets =  ImageNet_getMetaData
    
    metadatafile = '/lab/igpu3/u/jiaping/imageNet2010/devkit-1.0/data/meta.mat';    
    load(metadatafile, 'synsets');
    
end