numSubjects = length(event);

numVideoSegs = zeros(numSubjects,1);

segments = cell( numSubjects, 1 );
colorSegments = cell( numSubjects, 1 );

uEvent = [];

for i=1:numSubjects
    tmp_event = event{i};
    eventN0 = find( tmp_event ~= 0);
    subEvent = tmp_event(eventN0);
    
    uEvent = [uEvent ;unique(subEvent)];
    
    mark = 1;
    flag = true;
    iEvent = subEvent(1);
    colors = iEvent;
    
    for j=2:length(subEvent)
        if(subEvent(j) == iEvent)
            continue;
        else
            iEvent = subEvent(j);
            mark = [mark j];
            colors = [colors iEvent];
        end
    end
    
    mark = [mark length(subEvent)];
    subSegment = zeros(length(mark)-1,2);
    
    for j=1:length(mark)-1
        subSegment(j,:) = [mark(j) mark(j+1)];
    end
    
    segments{i} = subSegment;
    colorSegments{i}  = colors;
        
end

uEvent = unique(uEvent);
%% plot the data
style = {'r', 'g', 'b', 'c', 'y', 'm', 'k', 'w', '.k'};

map = zeros(max(uEvent),1);
for i=1:length(uEvent)
    map(uEvent(i)) = i;
end


for i=1:numSubjects
   
    subSegment = segments{i};
    colors = colorSegments{i};
    for j=1:size(subSegment,1)
        x = subSegment(j,:);
        y = [i i];
        plot(x',y', style{map(colors(j) )}, 'LineWidth', 10); hold on;
    end
end



% 3126
nSamples = [
         3126
         172
         104
          23
          39
          91
          39];
 nSampleRatio = nSamples;
 for i=2:7
     nSampleRatio(i) = nSampleRatio(i) + nSampleRatio(i-1);
 end

 nSampleRatio = nSampleRatio./sum(nSamples);
 
 nSampleRatio = [0 nSampleRatio'];
 nSampleRatio = nSampleRatio*10 + 1;
 
 segments = zeros(7,2);
 
  style = {'r', 'g', 'b', 'c', 'y', 'm', 'k'}; 
  
  hold on;
 for i=1:7
     segments(i,:) = [nSampleRatio(i) nSampleRatio(i+1)];
      x =  segments(i,:);
      y = [1.0 1.0];
     plot(x',y', style{i}, 'LineWidth', 10); hold on;
 end
  
 hold on;

 

a = 0;


