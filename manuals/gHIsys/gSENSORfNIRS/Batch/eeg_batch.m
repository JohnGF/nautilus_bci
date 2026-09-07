clear
clc
%Load Data
P_C=data;
P_C=load(P_C,uigetfile('','Select the calibration phase data.'));

%%
%%
%%
%Basic Parameters
%Trigger interval, 10Hz smapling Rate
sampBefore=2500;
sampAfter=3000;
%BL correction interval
CSPInterval=[sampBefore sampBefore+2000]; %3s befor the task onset
CSPFilters=[1 2 15 16];
P_C.SamplingFrequency=250;

%Feature Extraction Parameters
windowSize=750;
windowOverlap=725;

% Feature Matrix PArameters
FMInterval=[sampBefore/25 1 sampAfter/25+sampBefore/25];
%%
%%
%%

%Valid dongle
gbsanalyzeStrct.raw=P_C.Data;
%Select Trials and Channels
trial_id=[];
channel_id=[];
type_id=[];
channelnr_id=[2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17];
flag_tr='tr_exc';
flag_ch='ch_exc';
flag_type='type_exc';
flag_nr='nr_inc';
[TrialExclude, ChannelExclude]=gBSselect(P_C,trial_id,flag_tr,channel_id,flag_ch,type_id,flag_type,channelnr_id,flag_nr);


%Filter
Filter.Realization='butter';
Filter.Type='LP';
Filter.Order=6;
Filter.f_high=30;
FiltFiltFlag=0;
TrialExclude=[];
ChannelExclude=[1  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57];
P_C=gBSfilter(P_C,Filter,FiltFiltFlag,ChannelExclude,TrialExclude);


%Select Trials and Channels
trial_id=[];
channel_id=[];
type_id=[];
channelnr_id=[2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17];
flag_tr='tr_exc';
flag_ch='ch_exc';
flag_type='type_exc';
flag_nr='nr_inc';
[TrialExclude, ChannelExclude]=gBSselect(P_C,trial_id,flag_tr,channel_id,flag_ch,type_id,flag_type,channelnr_id,flag_nr);


%Filter
Filter.Realization='butter';
Filter.Type='HP';
Filter.Order=6;
Filter.f_low=8;
FiltFiltFlag=0;
TrialExclude=[];
ChannelExclude=[1  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57];
P_C=gBSfilter(P_C,Filter,FiltFiltFlag,ChannelExclude,TrialExclude);
gbsanalyzeStrct.filtered=P_C.Data;
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
channelnr_id=[2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17];
flag_tr='tr_exc';
flag_ch='ch_exc';
flag_type='type_exc';
flag_nr='nr_inc';
[TrialExclude, ChannelExclude]=gBSselect(P_C,trial_id,flag_tr,channel_id,flag_ch,type_id,flag_type,channelnr_id,flag_nr);


TrialExclude=[];
ChannelExclude=[1  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57];
P_C=gBScuttrialschannels(P_C,TrialExclude,ChannelExclude);

gbsanalyzeStrct.triggered=P_C.Data;
%CSP
clear T
Class1_nr=3;
Class2_nr=4;
T=CSPInterval; %from the task onset to 2s before the end
TrialExclude=[];
ChannelExclude=[];
[myFileName,myPath]=uiputfile('CSP_EEG.mat');
FileName=[myPath myFileName];
C_O=gBScsp(P_C,T,Class1_nr,Class2_nr,TrialExclude,ChannelExclude,FileName,0);




%Spatial Filter
SPF=spf;
Filter=load(SPF,FileName);
FilterNumber=CSPFilters;
Replace='replace all channels';
Transformation='Create temporal pattern';
P_C=gBSspatialfilter(P_C,Filter,FilterNumber,Replace,Transformation);


% Variance
ChannelExclude = [];
IntervalLength = windowSize;
GrowingWindow = 1;
Overlap = windowOverlap;
Replace = 'replace all channels';
FileName = '';
ProgressBarFlag = 0;
P_C = gBSvariance(P_C, ChannelExclude, IntervalLength, GrowingWindow,...
                  Overlap, Replace, FileName, ProgressBarFlag);
 
 
% Transform
ApplyOn = 'multiple channels';
ChannelExclude_mult = [];
TrialExclude_mult = [];
Operation_mult = 'NORM';
SecondOperand_mult(1) = 5;
Unit_mult = 'µV';
FirstOperand_two = 1;
Operation_two = 'SUB';
SecondOperand_two = [2];
ProgressBarFlag = 0;
P_C = gBSarithmetic(P_C, ApplyOn, ChannelExclude_mult,...
      TrialExclude_mult, Operation_mult, SecondOperand_mult,...
      Unit_mult, FirstOperand_two, Operation_two,...
      SecondOperand_two, ProgressBarFlag);
 
 
% Transform
ApplyOn = 'multiple channels';
ChannelExclude_mult = [];
TrialExclude_mult = [];
Operation_mult = 'LOG10';
SecondOperand_mult(1) = 5;
Unit_mult = 'µV';
FirstOperand_two = 1;
Operation_two = 'SUB';
SecondOperand_two = [2];
ProgressBarFlag = 0;
P_C = gBSarithmetic(P_C, ApplyOn, ChannelExclude_mult,...
      TrialExclude_mult, Operation_mult, SecondOperand_mult,...
      Unit_mult, FirstOperand_two, Operation_two,...
      SecondOperand_two, ProgressBarFlag);
 
gbsanalyzeStrct.FM=P_C.Data;
%Feature Matrix
Interval=FMInterval;
AttributeName={
    'CLASS2'
    'CLASS1'

};
ChannelExclude=[];
Permutate=0;
MergeTimePoints=0;
[myFileName,myPath]=uiputfile('FM_EEG.mat');
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
[myFileName,myPath]=uiputfile('EEG_Classifier.mat');
FileName=[myPath myFileName];
ProgressBarFlag=[0];
C_O=gBSlinearclassifier(F_M,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);

%get CSP weights
spf = get(Filter,'spf');
spf_struct = struct(spf);
W_CSP = spf_struct.D.W;
save W_CSP.mat W_CSP;