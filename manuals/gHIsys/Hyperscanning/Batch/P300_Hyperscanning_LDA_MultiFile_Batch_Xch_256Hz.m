% P300 Hyperscanning Speller Batch for gBSanalyze version 5.0
% g.tec medical engineering GmbH

% Use this batch for the data stored with gHyperscanning_ P300 models 
% run on postfixes: P300

global P_C              
global V_R

y=P_C.Data;             %import the data
y=squeeze(y)';

dataSampFreq = P_C.SamplingFrequency;
samplefreq = dataSampFreq / 4;
triallength=800;
NrOfChannels=size(P_C.Data,3)-3;

%% Downsample to 64 Hz
tmp=y;
clear y;

for ii=1:size(tmp,1)
    kk=2;
    for jj=1:4:size(tmp,2)-3
        meantmp=tmp(ii,jj:jj+3)';
        y(ii,kk)=mean(meantmp);
        kk=kk+1;
        
    end
end
y(10,1:length(y(10,:))-2)=y(10,3:length(y(10,:)));
y(11,1:length(y(11,:))-2)=y(11,3:length(y(11,:)));


%% Convert triallength from ms to samples
triallength=ceil(triallength*samplefreq/1000)+1;


%% Create Trialnumbers, increase Trialnumber when a Row/Column flashes
size_y=size(y);
trialnr=[];
max_trial=0;
for ii=1:size_y(2)-1
    if (y(size_y(1)-1,ii+1) > 0) && (y(size_y(1)-1,ii) == 0)
        max_trial=max_trial+1;
    end
    trialnr(ii+1)=max_trial;
end

%% Find out how often a Row or Column was intensified
trials=unique(trialnr);

%% Initialization of target arrays
index_withP300=1;
index_withoutP300=1;
withP300=[];
withoutP300=[];

% Transpose recorded data for compatibility isssues
y=y';

%% Bandpass Filter recorded Data

% No longer necessary, signal is already filtered

% Filter between 0.1 & 60 Hz for a samplerate of 240Hz
% Filter designed with sptool
% Butterworth Bandpass, Samplefreq. 240Hz
% Fstop1=0.01, Fpass1=0.1, Fpass2=30, Fstop2=119
% Astop1=40, Apass=1, Astop2=40
% load Filter.mat;
% signal_filtered=filter(Bandpass.tf.num, Bandpass.tf.den,...
%     y(:,2:NrOfChannels+1));

signal_filtered=y(:,2:NrOfChannels+1);


%% Extract Data from Bandpass filtered signal

% Define length of pre-stimulus-interval in ms
preStimulusms=100;
% Convert time in ms to samplenumber
preStimulus=ceil(preStimulusms*samplefreq/1000);

for cur_trial=min(trials)+1:max(trials)

    % get the indeces of the samples of the right trial
    trialidx=find(trialnr == cur_trial);

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
    for ii=1:size(trialdata,1)
        trialdata(ii,:)=trialdata(ii,:)-preStimulusOffset;
    end
        
    % Find out if current trial contains desired character
    % 0... row/column does not contain desired character
    % 1... intensified column does contain desired character
    cur_stimulustype=max(y(trialidx, size_y(1)));

    % If response to stimulus does not contain P300
    % save data to array withoutP300
    if cur_stimulustype == 0
        withoutP300.data(:,index_withoutP300*NrOfChannels-(NrOfChannels-1):...
            index_withoutP300*NrOfChannels)=trialdata;
        index_withoutP300=index_withoutP300+1;

    % If response to stimulus does contain P300
    % save data to array withP300
    else
        withP300.data(:,index_withP300*NrOfChannels-(NrOfChannels-1):...
            index_withP300*NrOfChannels)=trialdata;
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

%% Save data in correct format for LDA
% Each row in X is one feature vector
% For a feature vector the resulting data segments for each intensification
% are concatenated by channel
% Each row is the response to one stimulus

% Check if TrainClassifier File already exists
% If not create new variables and save file
if exist('AveragesForLDA_ClassifierData.mat') ~= 2
    X=train_LDA.X';
    K=train_LDA.Y';

    save AveragesForLDA_ClassifierData X K;
% If it already exist append data
else    
    load AveragesForLDA_ClassifierData.mat;
    AppendPos=size(X);
    AppendSize=size(train_LDA.X');
    X(AppendPos(1)+1:AppendPos(1)+AppendSize(1),:)=train_LDA.X';
    K(AppendPos(1)+1:AppendPos(1)+AppendSize(1),:)=train_LDA.Y';
    save AveragesForLDA_ClassifierData X K;
    
end

%% Create Classifier

global P300classifier;

P300classifier.method=1;
[P300classifier.F.weight,P300classifier.F.bias]=mlda_train_P300(X',K');
save P300classifier_LDA P300classifier

