
myDir = strcat('/Users/alejandroadriaquelozano/Documents/Systems Biology/Project 2/Results CI opt');
myFiles = dir(fullfile(myDir,'*.csv'));
cd '/Users/alejandroadriaquelozano/Documents/Systems Biology/Project 2'/'Results CI opt'/
myFiles=struct2table(myFiles);

combined_channels_fast=table();
for i=1:5
    T = readtable(myFiles.name{i});
    T= rows2vars(T);
    combined_channels_fast= vertcat(combined_channels_fast,T);
end

combined_channels_slow=table();
for i=6:10
    T = readtable(myFiles.name{i});
    T= rows2vars(T);
    combined_channels_slow= vertcat(combined_channels_slow,T);
end

y_all=split(combined_channels_fast.OriginalVariableNames,'_');
y_actors =  str2double(y_all(:,2));
y_emotions = y_all(:,1);
y= [y_actors ;y_emotions];
for i=1:length(y_emotions)
    if contains(y_emotions(i), 'a')
        y_emotions{i}='anger';
    elseif contains(y_emotions(i), 'd')
        y_emotions{i}='disgust';
    elseif contains(y_emotions(i), 'f')
        y_emotions{i}='fear';
    elseif contains(y_emotions(i), 'h')
        y_emotions{i}='happiness';
    elseif contains(y_emotions(i), 's')
        y_emotions{i}='sadness';
    end
end
y= table(y_emotions,y_actors);

combined_channels_fast.OriginalVariableNames=[];
channels_names= append('Channel',' ',string(1:98));
combined_channels_fast=renamevars(combined_channels_fast,combined_channels_fast.Properties.VariableNames,channels_names);

combined_channels_slow.OriginalVariableNames=[];
combined_channels_slow=renamevars(combined_channels_slow,combined_channels_slow.Properties.VariableNames,channels_names);

cd '/Users/alejandroadriaquelozano/Documents/Systems Biology/Project 2'/'Results CI opt'/
writetable(combined_channels_slow,'X_CI_opt_slow.csv')
writetable(combined_channels_fast,'X_CI_optfast.csv')
writetable(y,'y_CI.csv')

