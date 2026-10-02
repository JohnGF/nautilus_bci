
DataPath='C:\Users\Hintermueller\Documents\MATLAB\Final-Tests-August2012_1\Data_August';
FilePattern='*_param_estimate*.mat';
ignorepostfix = {'_raw','_conv','_unfiltered' '_unfilterd'};
NumRows = 6;
NumCols = 6;
Mode = 'RC' ; % 'RC' row column, 'SC' single character
ParadigmFrequency = 64;
TimeChannel = 1;
DefaultSamplingFrequency = 256;
files = dir(fullfile(DataPath,FilePattern));
for tc = 1:length(files)
    if    files(tc).isdir ...
       || any(cellfun(@(x)~isempty(x),regexp(files(tc).name,ignorepostfix,'once')))
        continue;
    end
     
    try
        fprintf(2,'Converting file ''%s'' ...',fullfile(DataPath,files(tc).name));
        tl = load(data,fullfile(DataPath,files(tc).name));        
        if ~isempty(TimeChannel)
            dt = tl.Data;        
            tl.SamplingFrequency = 1/min(unique(diff(squeeze(dt(:,:,TimeChannel)'))));
        else
            tl.SamplingFrequency = DefaultSamplingFrequency;
        end
    catch ME
        continue;
    end
    try
        [ ~ ] = convertP300DataToNewScheme(tl,NumRows,NumCols,Mode,ParadigmFrequency);
        fprintf(2,' Done\n');
    catch ME
        fprintf(2,' Failed\n');
    end
end