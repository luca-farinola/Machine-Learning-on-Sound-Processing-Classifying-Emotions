clear all ; close all ; clc 

em = 'anger';
dirPath = strcat('/Users/alejandroadriaquelozano/Documents/Systems Biology/Project 2/',em,'/');  % Directory path to the "anger" folder

% Get a list of all WAV files in the directory
fileList = dir(fullfile(dirPath, '*.wav'));

% Create an empty table to store the vectors
table = table();
disp(['CREATING GIANT MATRIX'])
for i = 1:length(fileList)
    %get name of the files in a folder 
    [~, fileName, ~] = fileparts(fileList(i).name);
    filename = strcat(dirPath,fileName,'.wav'); 
    disp(['running for file - ' fileName])
    % run model
    AC_Model()
    %Average across time 
    VectorEEresp1 = mean(EEresp1, 2);
    % Add the vector to the table with the filename as the column name
    table.(fileName) = VectorEEresp1;
end

% Assuming you have a table called 'myTable' that you want to save

% Specify the file name and path
filename = strcat(em,'.csv');

% Write the table to a CSV file 
 table= rows2vars(table);
 writetable(table, filename, 'Delimiter', ',');

 y= table.OriginalVariableNames;


