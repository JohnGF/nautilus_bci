clear;
clc;
%Load Data
P_C=data;
P_C=load(P_C,uigetfile('','Select the calibration phase data.'));

%%
%%
%%
%Basic Parameters
%Trigger interval, 10Hz smapling Rate
sampBefore=100;
sampAfter=120;
%BL correction interval
BLInterval=[70 100]; %3s befor the task onset
P_C.SamplingFrequency=250;

% fNIRS Feature Parameters
Mean = 1;
Moment = 0;
Slope = 1;
Complexity = 0;
Skewness = 0;
Kurtosis = 0;
PositiveArea = 0;
FrameSize = 30;

% Feature Matrix PArameters
FMInterval=[sampBefore 1 sampAfter+sampBefore];
%%
%%
%%


%Valid dongle

%Select Trials and Channels
trial_id=[];
channel_id=[];
type_id=[];
channelnr_id=[34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49];
flag_tr='tr_exc';
flag_ch='ch_exc';
flag_type='type_exc';
flag_nr='nr_inc';
[TrialExclude, ChannelExclude]=gBSselect(P_C,trial_id,flag_tr,channel_id,flag_ch,type_id,flag_type,channelnr_id,flag_nr);


%Filter
Filter.Realization='butter';
Filter.Type='LP';
Filter.Order=4;
Filter.f_high=0.3;
FiltFiltFlag=0;
TrialExclude=[];
ChannelExclude=[1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  50  51  52  53  54  55  56  57];
P_C=gBSfilter(P_C,Filter,FiltFiltFlag,ChannelExclude,TrialExclude);
gbsanalyzeStrct.filtered=P_C.Data;

% Down- Upsampling
NewSmpFreq = 10;
ProgressBarFlag = 0;
P_C = gBSdownupsampling(P_C, NewSmpFreq, ProgressBarFlag);
 
gbsanalyzeStrct.triggered=P_C.Data; 

%Trigger
New_tm{2}={53 1 'v' 0.9 0 'CLASS2' 'red'};
New_tm{1}={53 0 'v' -0.9 0 'CLASS1' 'blue'};
SamplesBefore=sampBefore;
SamplesAfter=sampAfter;
Uncomplete=0;
ChannelExclude=[];
P_C=gBStrigger(P_C,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);


%Select Trials and Channels
trial_id=[];
channel_id=[];
type_id=[];
channelnr_id=[34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49];
flag_tr='tr_exc';
flag_ch='ch_exc';
flag_type='type_exc';
flag_nr='nr_inc';
[TrialExclude, ChannelExclude]=gBSselect(P_C,trial_id,flag_tr,channel_id,flag_ch,type_id,flag_type,channelnr_id,flag_nr);


TrialExclude=[];
ChannelExclude=[1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  50  51  52  53  54  55  56  57];
P_C=gBScuttrialschannels(P_C,TrialExclude,ChannelExclude);

 
% Baseline Correction
Interval = BLInterval;
ChannelExclude = [];
TrialExclude = [];
ProgressBarFlag = 0;
P_C = gBSbaselinecorrection(P_C, Interval, ChannelExclude,...
                 TrialExclude, ProgressBarFlag);
 



FeaturesLabel = 'FNIRS';
ChannelExclude = [];
Replace = 'replace all channels';
FileName = '';
ProgressBarFlag = 0;
P_C = gBSfnirs(P_C, Mean, Moment, Slope, Complexity, Skewness, Kurtosis,...
                  PositiveArea, FrameSize, FeaturesLabel, ChannelExclude,...
                  Replace, FileName, ProgressBarFlag);
%Feature Matrix
Interval=FMInterval;
AttributeName={
    'CLASS2'
    'CLASS1'

};
ChannelExclude=[2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32];
Permutate=0;
MergeTimePoints=0;
[myFileName,myPath]=uiputfile('FM_HBO.mat');
FileName=[myPath myFileName];
ProgressBarFlag=[0];
F_M=gBSfeaturematrix(P_C,Interval,AttributeName,Permutate,MergeTimePoints,ChannelExclude,FileName,ProgressBarFlag);


%Load FeatureMatrix
F_M=featurematrix;
F_M=load(F_M,FileName);

%Linear Classifier
PlotFeatures=[1  2];
Method=['LDA'];
P.metric=[''];
TrainTestData=['100:100'];
[myFileName,myPath]=uiputfile('HBO_Classifier.mat');
FileName=[myPath myFileName];
ProgressBarFlag=[0];
C_O=gBSlinearclassifier(F_M,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);


%Feature Matrix
Interval=FMInterval;
AttributeName={
    'CLASS2'
    'CLASS1'

};
ChannelExclude=[1   3   5   7   9  11  13  15  17  19  21  23  25  27  29  31];
Permutate=0;
MergeTimePoints=0;
[myFileName,myPath]=uiputfile('FM_HBR.mat');
FileName=[myPath myFileName];
ProgressBarFlag=[0];
F_M=gBSfeaturematrix(P_C,Interval,AttributeName,Permutate,MergeTimePoints,ChannelExclude,FileName,ProgressBarFlag);


%Load FeatureMatrix
F_M=featurematrix;
F_M=load(F_M,FileName);

%Linear Classifier
PlotFeatures=[1  2];
Method=['LDA'];
P.metric=[''];
TrainTestData=['100:100'];
[myFileName,myPath]=uiputfile('HBR_Classifier.mat');
FileName=[myPath myFileName];
ProgressBarFlag=[0];
C_O=gBSlinearclassifier(F_M,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);