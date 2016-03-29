% this script is used to format the pre-trained alexnet 

function rgbd_format_pretrainedImageNetAlexNet
% format the pretrained AlexNet to our disentangling architecture

        arc.batchNormalization  = false;
        arc.conv.weightInitMethod = 'gaussian';
        arc.conv.scale          = 1.0;
        arc.conv.learningRate   = [1 2];
        arc.conv.weightDecay    = [1 0];
        arc.bnorm.learningRate  = [2 1];
        arc.bnorm.weightDecay   = [10 10];
        arc.norm.param          = [5 1 0.0001/5 0.75];
        arc.pooling.method      = 'max';
        arc.dropout.rate        = 0.5;
    
        bshare      =   true;
        bshare_lf   =   true;
        
        nclasses_obj            =  51;
        nclasses_transformation =  9;
        netInitParamFileName    =  'net_init_param.mat';
        saveDir = '/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-ImageNet-AlexNet/exp1';

        % initilize a disentangling architecture
        net_init_param = iLab_arc_de_dagnn_2streams_wL2_rndw(nclasses_obj, nclasses_transformation,...
                                      bshare, bshare_lf, arc, struct('isstructured', false, 'labelgraph', [])); 

        % used the pre-trained alexnet to initialize
        load(fullfile(saveDir, 'imagenet-matconvnet-alex.mat'));
        for p=1:10
            net_init_param.params(p).value = params(p).value;
        end
        
                                  
        net_init_param_ = net_init_param ;
        net_init_param  = net_init_param_.saveobj() ;
        save(fullfile(saveDir, netInitParamFileName), 'net_init_param') ;        
    
end
