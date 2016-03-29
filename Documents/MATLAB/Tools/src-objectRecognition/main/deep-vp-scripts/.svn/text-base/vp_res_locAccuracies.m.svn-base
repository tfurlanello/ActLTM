%% old vanishing point
olddist_ori = [ 1 120.55093502
  2 108.99893736
  3  94.79786056
  4  87.26186477
  5  82.81041967
  6  73.76463585
  7  68.50227807
  8  59.08175769
  9  57.58638780
 10  61.82754799
 11  84.33782714
 12 117.06308176
 13 146.94550677
 14 188.11802310
 15 188.99142164
 16  95.52636161
 17  84.76600645
 18  75.35803593
 19  68.77675206
 20  61.31923852
 21  56.70130612
 22  53.81568759
 23  45.69222856
 24  41.72747796
 25  42.04584953
 26  51.80749849
 27  85.36046242
 28 114.70084913
 29 159.25736497
 30 182.00099150
 31  78.91631603
 32  70.72286221
 33  52.50344171
 34  46.43259797
 35  41.24090935
 36  36.85941888
 37  33.41059350
 38  29.73182753
 39  28.49356363
 40  27.00819179
 41  27.83745416
 42  33.77565921
 43  55.87164533
 44 137.60319553
 45 171.31559040
 46  68.69811491
 47  61.48210404
 48  42.36846487
 49  38.25622618
 50  33.76502547
 51  30.68706963
 52  27.39474839
 53  25.88374608
 54  24.17833793
 55  21.57386061
 56  21.91736907
 57  28.39528543
 58  49.53202427
 59 125.38128545
 60 161.30665743
 61  62.49032384
 62  54.49312961
 63  38.47551325
 64  35.39207018
 65  31.35392303
 66  29.53973101
 67  27.39522884
 68  24.83063421
 69  23.15259703
 70  19.83967446
 71  21.08543598
 72  27.13220135
 73  45.19488157
 74 117.91095014
 75 153.48187094
 76  60.54991574
 77  54.53858713
 78  36.81247717
 79  34.70771444
 80  31.11294932
 81  30.10150019
 82  27.52577529
 83  24.87435946
 84  22.84773758
 85  20.11468286
 86  21.68049158
 87  26.04738067
 88  45.17462791
 89 106.63513588
 90 146.61515226
 91  59.09632019
 92  54.04454421
 93  36.94887973
 94  34.28273943
 95  30.95184878
 96  29.64884246
 97  26.85317595
 98  24.90609256
 99  22.65793177
100  20.38726079
101  22.35660940
102  26.61384669
103  43.45132836
104  97.21882621
105 139.56985843
106  61.66891130
107  55.23327586
108  36.74140676
109  34.93415392
110  32.27681961
111  30.08951561
112  27.49609946
113  25.25262888
114  23.09594359
115  20.65760276
116  23.44471691
117  28.19995420
118  42.34772129
119  86.72615178
120 136.94644572
121  59.31602230
122  54.91440855
123  36.65448373
124  35.27188668
125  31.57227311
126  30.00688041
127  27.82609978
128  25.77509557
129  22.21922125
130  20.24926920
131  23.31718553
132  29.38687159
133  42.26144752
134  85.50571094
135 133.05292539
136  59.71960590
137  56.17034349
138  36.83615327
139  34.57439094
140  32.12227071
141  29.78592010
142  27.67449579
143  24.95401192
144  23.13667226
145  20.64829795
146  21.66002111
147  25.22863916
148  43.30275801
149  87.65160908
150 135.91502102
151  60.53981826
152  56.12674203
153  37.74633562
154  35.64173469
155  32.40540868
156  30.84568138
157  28.82666472
158  26.37646732
159  24.35841057
160  21.34238413
161  20.88137349
162  24.62934615
163  41.70463732
164  95.80281913
165 137.03863685
166  66.79752676
167  62.83626224
168  41.57435468
169  39.12860544
170  36.02872585
171  34.08776873
172  31.85751874
173  29.39499576
174  27.49667816
175  24.77416691
176  23.16366276
177  25.57565199
178  46.99184395
179 106.53476957
180 147.99947444
181  80.85295833
182  72.00999910
183  51.20916866
184  46.02445479
185  43.40970672
186  40.10974282
187  37.72668761
188  35.06253043
189  33.54210992
190  31.63909326
191  30.54900545
192  34.40450779
193  59.52463349
194 121.67687831
195 156.42686983
196  97.16957753
197  90.53631262
198  83.95684895
199  76.97048109
200  69.02882809
201  65.50444659
202  61.11059161
203  58.42238123
204  55.25514645
205  52.99947698
206  56.11582590
207  62.44749681
208  91.64808183
209 143.31867556
210 164.11820464
211 123.41159447
212 115.94719405
213 105.46756110
214  97.82294714
215  91.41991117
216  85.61920734
217  80.68196829
218  72.12077713
219  68.37275774
220  69.77020303
221  73.24570900
222  90.27099279
223 132.94406478
224 163.55334537
225 172.13193896];

nclasses = 15;
olddist = olddist_ori(:,2);

[idx_ori, c] = kmeans(olddist, nclasses);
[~, c_idx] = sort(c);
idx = idx_ori;

for i=1:numel(c_idx)
   ref = c_idx(i);
   idx(idx_ori == ref) = i;   
end

figure; imagesc(reshape(idx, [15 15]));




resSaveDir = '/lab/igpu3/u/kai/deep_vp/results/iros2016';
%% deep vanishing point
resDir = '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc1024';
resFileName = 'prediction-prob.mat';

accFile = fullfile(resDir, resFileName);
load(accFile);

mapping_label2coord = '/lab/igpu3/u/kai/deep_vp/results/google_dataset/vp-alexnet-simplenn-obj/mapping.mat';
load(mapping_label2coord);

locAccuracies  =  zeros(1,225);
locDist        =  zeros(1,225);

for i=1:225    
    b_gt = (labels_gt == i);
    b_pred = (labels_pred(:,1) == i);    
    b = b_gt & b_pred;    
    locAccuracies(i) = sum(b) / sum(b_gt);
    
    % distance deviations

    refx = mapping(labels_gt(b_gt),2);
    refy = mapping(labels_gt(b_gt),3);

    pred = labels_pred(:,1);
    predx = mapping(pred(b_gt), 2);
    predy = mapping(pred(b_gt), 3);

    locDist(i) = mean(sqrt( (predx - refx).^2 + (predy - refy).^2));
    
end

locAcc_map  =reshape(locAccuracies, [15  15]);


[idx_ori, c] = kmeans(locDist(:), nclasses);
[~, c_idx] = sort(c);
idx = idx_ori;

for i=1:numel(c_idx)
   ref = c_idx(i);
   idx(idx_ori == ref) = i;   
end

figure; imagesc(reshape(idx, [15 15]));


%% combined
nclasses = 100;
scale = 5;
combinedDist = [olddist(:); locDist(:)];
olddist_re = imresize(reshape(olddist(1:225), [15 15]), scale);
locDist_re = imresize(reshape(locDist(1:225), [15 15]), scale);
% combinedDist_re = [reshape(combinedDist(1:225), [15 15]), reshape(combinedDist(226:end), [15 15])];
% combinedDist_re = imresize(combinedDist_re, scale);
% combinedDist_re = combinedDist_re(:);
combinedDist_re = [olddist_re locDist_re];


[idx_ori, c] = kmeans(combinedDist_re(:), nclasses);
[c, c_idx] = sort(c);
idx = idx_ori;

for i=1:numel(c_idx)
   ref = c_idx(i);
   idx(idx_ori == ref) = i;   
end

img_display = [reshape(idx(1:225*scale^2), [15*scale 15*scale]), reshape(idx((225*scale^2+1):end), [15*scale 15*scale])];
oldidx_re = imresize(reshape(idx(1:225*scale^2), [15*scale 15*scale]), 2);
newidx_re = imresize(reshape(idx((225*scale^2+1):end), [15*scale 15*scale]), 2);
img_display2 = [oldidx_re newidx_re];
figure; imagesc(img_display);
figure; imagesc(img_display2);
axis equal; axis tight;
set(gca, 'xtick', [], 'ytick', []);

ticks = [1:10:nclasses nclasses];
labels = {};
for i=1:numel(ticks)
    labels = [labels; sprintf('%.0f',c(ticks(i)))];
end

colormap jet;
colorbar('Ticks', ticks,...
         'TickLabels',labels, 'fontsize', 25);

export_fig(fullfile(resSaveDir, 'location-dependent-accuracy.png'), '-png', '-m3', gcf);
export_fig(fullfile(resSaveDir, 'location-dependent-accuracy.pdf'), '-pdf', '-m1', '-transparent', gcf);
export_fig(fullfile(resSaveDir, 'location-dependent-accuracy.eps'), '-eps', '-m1', '-transparent', gcf);

% mesh
[x,y] = meshgrid(1:size(oldidx_re,1));
figure; 
mesh(x,y, oldidx_re); hold on; mesh(x,y, newidx_re);

figure; 
[x,y] = meshgrid(1:size(olddist_re,1));
mesh(x,y, olddist_re); hold on; mesh(x,y, locDist_re);
zlabel('VP prediction errors (pixels)', 'fontsize', 30);
set(gca, 'fontsize', 15);
title('location-dependent VP prediction comparison', 'fontsize' ,30);
% figure; imagesc([reshape(idx(1:225), [15 15]), ones(15,15)]);
% daspect([1,1,.3]);

OptionZ.FrameRate=25;OptionZ.Duration=10; OptionZ.Periodic=true;
CaptureFigVid([-84 0; 20,0; 110, 20; 190, 15; 290,0; 380,3], 'WellMadeVid',OptionZ)


figure; imagesc([reshape(combinedDist(1:225), [15 15]), reshape(combinedDist(226:end), [15 15])]);

