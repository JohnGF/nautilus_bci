function loadbci2000header;
global P_C

%open file
t=fopen(P_C.FileName);
if t < 0 
    error('File ''%s'' not found or no data not recorded by bci2000',P_C.FileName)
end
%look how long it is
fseek(t, 0, 'eof');
filesize = ftell(t);
fseek(t, 0, 'bof');
DataFormat='short'; %default data format
%first line: 
%HeaderLen= 8834  SourceCh= 64 StatevectorLen= 9
%get header length
tmp=fgetl(t);
idx0=strfind(tmp,'BCI2000V= ');
idx1=strfind(tmp,'HeaderLen= ');
idx2=strfind(tmp,'SourceCh= ');
idx3=strfind(tmp,'StatevectorLen= ');
idx4=strfind(tmp,'DataFormat= ');

Version=str2num(tmp( idx0+10:idx1-1 ) );
HeaderLen=str2num(tmp( idx1+11:idx2-1 ) );
SourceCh=str2num(tmp( idx2+10:idx3-1 ) );
if Version==1.1
    StatevectorLen=str2num(tmp( idx3+16:idx4-1 ) );
    DataFormat=(tmp (idx4+12:length(tmp) ) );
else
    StatevectorLen=str2num(tmp( idx3+16:length(tmp) ) );
end

while 1
    
    tmp=fgetl(t);
    %search for end of header
    if strcmp(tmp,''), 
        break, 
    end
    
    %search the sampling rate
    idx=(strfind(tmp,'SamplingRate='));
    if isempty(idx)==0
        tmp=tmp(idx+14:end);
        idx1=strfind(tmp,' ');
        smpf=str2num(tmp(1:idx1(1)));
        
    end
end

%total number of samples
%total number of samples
if strcmp(DataFormat,'float32')
    samples=(filesize-HeaderLen)/((4*SourceCh)+StatevectorLen);
else
    samples=(filesize-HeaderLen)/((2*SourceCh)+StatevectorLen);
end

%reserve memory and read in the data
Trigs=zeros(samples,2);
B=zeros(samples,StatevectorLen);
for i=1:samples
    tmpdata=fread(t,[(SourceCh)],DataFormat);
    tmp=fread(t,[StatevectorLen],'char');
    str=[];
    for ii=StatevectorLen:-1:1
        str=[str dec2bin(tmp(ii),8)];
    end
    len=length(str)+1;

    %TargetCode
    Byte=2; Bit=4; Len=5;
    Trigs(i,1)=bin2dec (str(len-(Byte*8+Bit+1+Len-1):len-(Byte*8+Bit+1)));
    
    %Feedack
    Byte=6; Bit=6; Len=2;
    Trigs(i,2)=bin2dec (str(len-(Byte*8+Bit+1+Len-1):len-(Byte*8+Bit+1)));
end
fclose(t);

%define the markers
marker=P_C.Marker;
markername=P_C.MarkerName;
markercolor=P_C.MarkerColor;

tmp1=[find(diff(Trigs(:,1))>0)];
markername=[markername; 'TargetCode1'];
markercolor=[markercolor; 'green'];

for i=1:length(tmp1)
    tmp=[tmp1(i) 1 3];
    marker=[marker; tmp];

end

P_C.Marker=marker;
P_C.MarkerColor=markercolor;
P_C.MarkerName=markername;


%create file
classes=zeros( length(tmp1),2);
for i=1:length(tmp1)
    if Trigs(tmp1(i)+1,1)==1
        classes(i,1)=1;
    else
        classes(i,2)=1;
    end
end
save classes classes

