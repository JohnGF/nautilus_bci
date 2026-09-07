function P_C = convertP300DataToNewScheme(P_C,NumRows,NumCols,mode,parafrequ)
% SYNOPSIS: P_C = convertP300Data(P_C,NumRows,NumCols,mode,parafrequ)
% 
% P_C .......... input P_C object
% NumRows ...... Number of rows of symbol matrix
% NumCols ...... Number of columns of symbol matrix
% mode ......... Mode used to flash the matrix one of
%                2, 'SC': for single character mode (case insensitive)
%                3, 'RC': for row collumn mode (case insensitive)
% parafrequ .... Sampling frequency the Paradigm was operating at
%                for data recorded using g.BCI, g.BCI_XML_standard and 
%                g.BCI_SOCI models use 64Hz


y=squeeze(P_C.Data)';
fs = P_C.SamplingFrequency;
patterns=y(size(y,1)-1,:);
targets = find(patterns > 0);
if isempty(targets)
    error('not supported probably trying to convert new data set with pattern already activated')
end
breaks = diff(targets);
[ spacing spaceid ] = sort(breaks);
%spacing = medfilt1(spacing,5);
detect = diff(spacing);
here = find(detect > max(detect) * 0.1) + 1;
breakloc = spaceid(here:end);
breakstarts = targets(breakloc);
breakends = [ targets(1) - 1 breakstarts + spacing(here:end)-1 ]  ;

npeaksamples = fs / parafrequ;
if floor(npeaksamples) ~= npeaksamples
    error('parafrequ not multiple of sampling rate');
end
if isnumeric(mode)
    mode = abs(mode);
end
switch mode
    case { 2,'SC','sc','Sc' ,'sC' }        
        patterns(2,:) = patterns;
        patterns(1,targets) = 1;
        eots = zeros(1,length(targets));        
        nflashes = NumRows * NumCols;
        eots(nflashes:nflashes:length(targets)) = 1;
        patterns(3,targets) = eots;
    case { 3, 'RC','rc','rC', 'Rc' }        
        targetvals = patterns(targets);
        symbolcount =  max([NumCols,NumRows])+1;
        patterns(1:symbolcount,:) = zeros(symbolcount,length(patterns));
        ids = 1:(NumRows * NumCols);
        matrix = reshape(ids,[NumCols NumRows])';
        for n=1:length(targetvals)
            if targetvals(n) > NumCols
                patterns(:,targets(n)) = [ NumCols matrix(targetvals(n) - NumCols,:) ]';
            else
                patterns(:,targets(n)) = [ NumRows; matrix(:,targetvals(n)) ];
            end
        end
        nflashes = NumCols + NumRows;
        eots = zeros(1,length(targets));
        for put = nflashes * npeaksamples + [ -npeaksamples + 1:0]
            eots(put:nflashes*npeaksamples:length(eots)) = 1;
        end
        patterns(end+1,targets) = eots;
    otherwise
        error('not supported probably trying to convert new data set with pattern already activated')
end
breakends(end+1:end+length(breakends)) = breakends + 2;
breakends(end+1:end+length(breakends)) = breakends + 1;
initline = zeros(1,size(y,2));
initline(-4 * npeaksamples + sort(breakends) + 1) =  NumRows*NumCols;
if mode > 0
    y = [ y(1:end-2,:);initline;patterns;y(end,:) ] ;
else
    initline(-4 * npeaksamples + sort(breakends) + 1 -npeaksamples) =  2;
    initline(-4 * npeaksamples + sort(breakends) + 1 - 2 * npeaksamples ) = 0;
    initline(-4 * npeaksamples + sort(breakends) + 1 - 3 * npeaksamples ) =  abs(mode);
    initline(-4 * npeaksamples + sort(breakends) + 1 - 4 * npeaksamples ) =  1;
    y = [ y(1:end-2,:);initline;y(end-1:end,:) ] ;
end
[ cpath cname cext ] = fileparts(P_C.FileName);
convert = fullfile(cpath,[cname '_conv' cext]);
save(convert,'y');
P_C = load(data,convert);
P_C.SamplingFrequency = fs;


