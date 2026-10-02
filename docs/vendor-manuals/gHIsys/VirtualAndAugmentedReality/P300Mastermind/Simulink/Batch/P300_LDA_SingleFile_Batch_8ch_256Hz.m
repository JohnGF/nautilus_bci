
keeponleave = who();
% fallback frequency for downsampled data, in case not deduceable from P_C
% object SamplingFrequency or the TimeChannel recorded along with the data
samplefreq=64;
downsamplefactor = 4; % factor by which to downsample the data relative to P_C.Sampling frequency
TrialLength=800;
NrOfChannels=8;
TimeChannel = 1; % channel encoding the sample time for each sample since start of data set
TrainingData = 50; % [%]
TestData = 50 ; % [%]
NumFlashes = 15; % Num repetitions
ResidualError = 0.45;
ExplicitTarget = false;

if ~exist('P_C','var')
    global P_C
end
% global V_R

NReps = 25;
fname = P_C.FileName;
y=squeeze(P_C.Data)';             %import the data


%% Downsample to 64 Hz
tmp=y;
clear y;
if isempty(P_C.SamplingFrequency)
    if    ~isempty(TimeChannel) ...
       && ( floor(TimeChannel) == TimeChannel ) ...
       && ( TimeChannel > 0 ) ...
       && ( TimeChannel <= size(tmp,1) )
        timeintervals = unique(diff(tmp(TimeChannel,:)));
        [ interval, count ] = min(timeintervals);
        if    ( interval <= 0 ) ...
           || ( count > 1 ) 
            P_C.SamplingFrequency = samplefreq * downsamplefactor;
        else         
            P_C.SamplingFrequency = 1/ timeintervals(1);
        end
    else
        P_C.SamplingFrequency = samplefreq * downsamplefactor;
    end
end
samplefreq = P_C.SamplingFrequency / downsamplefactor;
stopat = downsamplefactor - 1;
for ii=1:size(tmp,1)
    kk=2;
    for jj=1:downsamplefactor:size(tmp,2)-stopat
        meantmp=tmp(ii,jj:jj+stopat)';
        y(ii,kk)=mean(meantmp);
        kk=kk+1;
        
    end
end
tgline = size(y,1);
initline = NrOfChannels + ( TimeChannel > 0 ) + 1;
fflline = initline + 1;
lflline = tgline - 1;
if fflline > lflline
    error('Improper,outdated data set: Probably init line missing.\nCall P_C=convertP300Data(P_C); first');
end
y(fflline:lflline,1:length(y(fflline,:))-2)=y(fflline:lflline,stopat:length(y(fflline,:)));
y(tgline,1:length(y(tgline,:))-2)=y(tgline,stopat:length(y(tgline,:)));
y(initline,1:length(y(initline,:))-2)=y(initline,stopat:length(y(initline,:)));

keeponleave(end+1:end+8) = {'tgline','initline','fflline','lflline','P300Classifier','NumSym','stat','TrialLength'};

modesettings=find(y(initline,:)==1);
if ~isempty(modesettings)
    error('File: ''%s'' probably contains old P300 training data format, please use ''convertP300DataToNewScheme'' function to convert it first',P_C.FileName);
    
else
    modesettings = find(y(initline,:)>0);
    if ~any(y(fflline,modesettings(1):modesettings(2))>0)
        modesettings(1) = [];
    end
    modesettings(end+1) = size(y,2);
    collect = length(modesettings)-1;    
    mode = 4;
    noeots = 0;
    invalideots = 0;
    ignoretrial = false(1,collect);
    for ckt = 1:collect
        seentrials = y(lflline,modesettings(ckt):modesettings(ckt+1));
        counttrials = find(seentrials>0);
        if length(counttrials) ~= NumFlashes
            ignoretrial(ckt) = true;
            if y(tgline,modesettings(ckt))>0
                if y(tgline,modesettings(ckt+1)<1)
                    y(tgline,modesettings(ckt+1)) = y(tgline,modesettings(ckt));
                end
            end
            y([initline:tgline],modesettings(ckt):modesettings(ckt+1)-2) = 0;
            continue;
        end
        y(fflline:lflline,(modesettings(ckt) + counttrials(end)+1):(modesettings(ckt+1)-2)) = 0;
    end
    modesettings(ignoretrial) = [];
    collect = length(modesettings)-1;
    numtargets = length(modesettings)-1;
    if collect < 1
        error('dataset without completed sequences not supported');
    end
    seentrials = y(lflline,modesettings(1):modesettings(2));
    counttrials = find(seentrials>0);
    if ( max(seentrials)> 1)
        error('dataset without end of trial triggers not supported');
    elseif ~any(seentrials)
        error('dataset without end of trial triggers not supproted')
    end        
    NumSym = y(initline,modesettings(1));
    counttrials = [ modesettings(1) ,  counttrials + modesettings(1) ] ;
    patcount = zeros(1,length(counttrials));
    for cntpat = 1:( length(counttrials) - 1 )
        patties = find(y(fflline,counttrials(cntpat)+1:counttrials(cntpat+1))>0);
%         any(diff(patties)<2);
        patcount(cntpat) = length(patties) * NumFlashes;
    end
    patcount(end) = [];
    patcount = min(patcount);
    for cm = 1:length(modesettings)-1
        patloc = find(y(fflline,modesettings(cm):modesettings(cm+1))>0);
        try
            patloc(1:patcount)=[];
            if isempty(patloc)
                continue;
            end
            y(fflline:tgline,patloc + modesettings(cm)-1) = 0;
        catch ME
            warning('not enough patterns for symbol %d',cm);
        end
    end
    modesettings(end) = [];
end

targets = y(tgline,modesettings);
if isempty(targets)
    if isempty(find(y(tgline,:) > 0,1,'First'))
        error('Free run data set without targets');
    end
end
%% Convert triallength from ms to samples
triallength=ceil(TrialLength*samplefreq/1000)+1;


%% Create Trialnumbers, increase Trialnumber when a Row/Column flashes
size_y=size(y);
trialnr=zeros(1,size(y,2));
allpatterns = find(y(fflline,:)>0);
max_trial = length(allpatterns);
patterns = y(fflline:lflline,allpatterns);
allpatterns(end+1) = size(y,2);
for tr = 1:length(allpatterns) - 1
    trialnr(allpatterns(tr):allpatterns(tr+1)) = tr;
end

if length(mode) == 1 && mode == 4 
    collect = max_trial / NumFlashes / collect ;
    if ( round(collect) ~= collect )
        error('Number of trials and number of flashes per trial must be consistent')
    end
end
%% Find out how often a Row or Column was intensified
trials=unique(trialnr);
trials(trials==0) = [];
if length(trials) ~= size(patterns,2)
    error('inconsitent flash pattern detected, update your model or convert data first');
elseif ( size(patterns,1) < 2 ) 
    symbolmap = unique(patterns,'sorted')';
    if isempty(mode)
        patterns = [ones(1,1:length(patterns));patterns];
        NumFlashes = max(symbolmap);
    elseif max(symbolmap) ~= mode(1)+mode(2)
        error('row column mode matrix dimensions and flash patterns do not match');
    else
        symmatrix = reshape(1:mode(1)*mode(2),[mode(2),mode(1)])';
        symbolmap = zeros(mode(1)+mode(2),max(mode)+1);
        for r=1:mode(1)
            symbolmap(r+mode(2),1:mode(2)+1) = [ mode(2), symmatrix(r,:)];
        end
        for c=1:mode(2)
            symbolmap(c,1:mode(1)+1) = [ mode(1) symmatrix(:,c)'];
        end
        patterns=symbolmap(patterns,:)';
        
    end
else
    % ignore mode when data set stores flash patterns.
    % the first flash line indicates how many of the following flash lines
    % encode the pattern used. Thus for the simpliest pattern the single
    % character pattern lfflline - fflline == 1 must hold. in any other
    % case it is an old data set with single character or row column
    % flashes.    
end
symmatrix = zeros(7,max(max(patterns)));

%% Initialization of target arrays
index_withP300=1;
index_withoutP300=1;
withP300=[];
withoutP300=[];
unsortWithP300=[];
unsortWithoutP300=[];

% Transpose recorded data for compatibility isssues
y=y';

takechannels = 1:NrOfChannels+(TimeChannel>0);
takechannels(TimeChannel) = [];
%% Bandpass Filter recorded Data

% No longer necessary, signal is already filtered
% if reverted uncommetn block below

% Filter between 0.1 & 60 Hz for a samplerate of 240Hz
% Filter designed with sptool
% Butterworth Bandpass, Samplefreq. 240Hz
% Fstop1=0.01, Fpass1=0.1, Fpass2=30, Fstop2=119
% Astop1=40, Apass=1, Astop2=40
% load Filter.mat;
% signal_filtered=filter(Bandpass.tf.num, Bandpass.tf.den,...
%     y(:,takechannels));

%% comment this line when data should again be filtered
signal_filtered=y(:,takechannels);


%% Extract Data from Bandpass filtered signal

% Define length of pre-stimulus-interval in ms
preStimulusms=100;
% Convert time in ms to samplenumber
preStimulus=ceil(preStimulusms*samplefreq/1000);


curtartget = nan;
nexttarget = nan;
if ~isempty(targets)
    curtarget = targets(1);
    nexttarget = 2;
end

for cur_trial=min(trials):max(trials)

    % get the indeces of the samples of the right trial
    trialidx=find(trialnr == cur_trial);

    if nexttarget < length(modesettings) && trialidx(1) > modesettings(nexttarget);
        curtarget = targets(nexttarget);
        nexttarget = nexttarget + 1;
    end        
        
    % extract data for response to each intensification
    % extraction starts at the beginning of each intensification
    % data for the length of the time window is extracted
    trialdata=...
        signal_filtered(min(trialidx)+1:min(trialidx)...
        +triallength-preStimulus-1,:);
    
    % extract pre-stimulus-interval
    preStimulusData=...
        signal_filtered(min(trialidx)-preStimulus+2:...
        min(trialidx),:);
    % average pre-stimulus-interval
    
    preStimulusOffset=mean(preStimulusData);
    
    % Perform offset correction
    for ii=1:length(trialdata)
        trialdata(ii,:)=trialdata(ii,:)-preStimulusOffset;
    end
    
    if ~isnan(curtarget)
        if any(y(trialidx(1),fflline+1:lflline-1)==curtarget)
            cur_stimulustype = 1;
        else
            cur_stimulustype = 0;
        end
    else
        % Find out if current trial contains desired character
        % 0... row/column does not contain desired character
        % 1... intensified column does contain desired character
        cur_stimulustype=max(y(trialidx, tgline));
    end

    % If response to stimulus does not contain P300
    % save data to array withoutP300
    if cur_stimulustype == 0
        withoutP300.data(:,index_withoutP300*NrOfChannels-(NrOfChannels-1):...
            index_withoutP300*NrOfChannels)=trialdata;
        unsortWithoutP300(index_withoutP300) = cur_trial;
        index_withoutP300=index_withoutP300+1;

    % If response to stimulus does contain P300
    % save data to array withP300
    else
        withP300.data(:,index_withP300*NrOfChannels-(NrOfChannels-1):...
            index_withP300*NrOfChannels)=trialdata;
        unsortWithP300(index_withP300) = cur_trial;
        index_withP300=index_withP300+1;
    end
end

%% Moving average filtering of extracted data
windowSize = 3;
withP300.filtered=filter...
    (ones(1,windowSize)/windowSize,1,withP300.data);
withoutP300.filtered=filter...
    (ones(1,windowSize)/windowSize,1,withoutP300.data);

%% Downsample data
withP300.downsampled=downsample(withP300.filtered, windowSize);
withoutP300.downsampled=downsample(withoutP300.filtered, windowSize);

%% Create data vectors for LDA
train_LDA=[];
size_withP300=size(withP300.downsampled);
size_withoutP300=size(withoutP300.downsampled);
train_LDA.X=zeros(size_withP300(1)*NrOfChannels,...
     size_withP300(2)/NrOfChannels+size_withoutP300(2)/NrOfChannels);

%% Write vectors for trainingdata with P300 response
for ii=1:size_withP300(2)/NrOfChannels
    for kk=1:NrOfChannels
        train_LDA.X(kk*size_withP300(1)-(size_withP300(1)-1):...
            kk*size_withP300(1),ii)=...
            withP300.downsampled(:,(ii-1)*NrOfChannels+kk);        
        kk=kk+1;
    end
    train_LDA.Y(ii)=1; % Class label is 1 if signal contains P300
    ii=ii+1;
end


%% Append vectors for trainingdata without P300 response
for ii=1:size_withoutP300(2)/NrOfChannels
    for kk=1:NrOfChannels
        train_LDA.X(kk*size_withoutP300(1)-(size_withoutP300(1)-1):...
            kk*size_withoutP300(1),ii+size_withP300(2)/NrOfChannels)=...
            withoutP300.downsampled(:,(ii-1)*NrOfChannels+kk);        
        kk=kk+1;
    end
    train_LDA.Y(ii+size_withP300(2)/NrOfChannels)=2;
    % Class label is 2 if signal does not contain P300
    ii=ii+1;
end

%% Create Classifier
X=train_LDA.X';
K=train_LDA.Y';

P300Classifier = generateMLDAClassifier_probability(X,K);
[ pth nm ex ] = fileparts(fname);
if ~exist('P300classifier_LDA','var')    
    P300classifier_LDA = fullfile(pth,[nm '_LDA.mat']);
end
P300Classifier.FileName = P300classifier_LDA;

compute = struct();
compute.X = X;
compute.K = K;
compute.target = unsortWithP300;
compute.nontarget = unsortWithoutP300;
compute.max_trial = max_trial;
compute.NReps = NReps;
compute.symmatrix = symmatrix;
compute.NumFlashes = NumFlashes;
compute.patterns = patterns;
compute.collect = collect;
compute.NumSym = NumSym;
compute.DefaultResidualError = ResidualError;
[fighnd, P300Classifier, stat ] = calculateStatistics_probability(P300Classifier,ResidualError,true,compute);
save(P300classifier_LDA,'P300Classifier');

P300classifier_stat = fullfile(pth,[nm '_stat.mat']);
save(P300classifier_stat,'stat');
print(fighnd,regexprep(P300classifier_stat,'\.mat$','.png'),'-dpng','-r200','-loose','-noui');
killonleave = setdiff(who(),keeponleave);
clear(killonleave{:});
