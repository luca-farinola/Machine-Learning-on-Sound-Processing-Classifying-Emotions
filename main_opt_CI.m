clear all ; close all ; clc 

em = 'disgust';
dirPath = strcat('/Users/alejandroadriaquelozano/Documents/Systems Biology/Project 2/',em,'/');  % Directory path to the "anger" folder

% Getting data for CI simulation
selected_channels_slow= readtable('important_slow.txt');

selected_channels= selected_channels_slow.Var1;
%%
% Get a list of all WAV files in the directory
fileList = dir(fullfile(dirPath, '*.wav'));

% Create an empty table to store the vectors
table_channels_fast = table();
table_spectrotemporal_fast = table();
table_channels_slow = table();
table_spectrotemporal_slow = table();
disp(['CREATING GIANT MATRIX'])
for i = 1:length(fileList)
    %get name of the files in a folder 
    [~, fileName, ~] = fileparts(fileList(i).name);
    filename = strcat(dirPath,fileName,'.wav'); 
    disp(['running for file - ' fileName])
    % run model
    AC_Model()

    %Average across time FAST
    VectorEEresp4 = mean(EEresp4, 2);
    % Add the vector to the table with the filename as the column name
    table_channels_fast.(fileName) = VectorEEresp4;

    %Average across time SLOW
    VectorEEresp3 = mean(EEresp3, 2);
    % Add the vector to the table with the filename as the column name
    table_channels_slow.(fileName) = VectorEEresp3;

    
end

cd /Users/alejandroadriaquelozano/Documents/'Systems Biology'/'Project 2'/'Results CI opt'/
 writetable(table_channels_fast, 'table channels CI fast disgust.csv', 'Delimiter', ',');

 writetable(table_channels_slow, 'table channels CI slow disgust.csv', 'Delimiter', ',');
