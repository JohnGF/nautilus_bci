
%Batch for SSVEP Classification with Minimum Energy
% run on postfixes: SSVEP
%##################################################

global P_C
global V_R

%save Classifier in Your Installation path:\gSSVEP_8ch\train_lda_session1.mat
path = mfilename('fullpath');
idx = strfind(path, '\');
idx = idx(end-1);
path = path(1:idx);
save_path = [ path 'train_lda_session1.mat' ];

clear idx path 

sampFrequency = P_C.SamplingFrequency;

%Classinfo File
%--------------
classinfo = 'classinfo_20tr.m';

%Stimulation frequencies
%-----------------------
frequencies = [10  11  12  13];

%Trigger-settings
%------------------
TriggerLow = 3;
TriggerHigh = 7;
SamplesBefore = round(TriggerLow*sampFrequency);   %3s
SamplesAfter = round(TriggerHigh*sampFrequency);   %7s

%Minimum Energy Settings
%-------------------------
NumberOfHarmonics = 1;
EvaluationStep = round(0.2*sampFrequency);  % 200ms
IntervalLength = round(3*sampFrequency);     % 3s
ModelOrder = 7;
ReestimateFreq = 5;

%Feature Matrix Settings
%-------------------------
if sampFrequency == 250
    Interval=[1   3  ceil((TriggerLow+TriggerHigh)*ReestimateFreq)];
else
    Interval=[1   3  ceil((TriggerLow+TriggerHigh)*ReestimateFreq+1)];
end

old_PC = P_C;

try
    %Select Trials and Channels
    trial_id=[];
    channel_id=[];
    type_id=[];
    channelnr_id=[1  10 11];
    flag_tr='tr_exc';
    flag_ch='ch_exc';
    flag_type='type_exc';
    flag_nr='nr_exc';
    [TrialExclude, ChannelExclude]=gBSselect(P_C,trial_id,flag_tr,channel_id,flag_ch,type_id,flag_type,channelnr_id,flag_nr);
    
    %Trigger
    New_tm{1}={10 1 'l' 90 0 'TRIG' 'red'};
    Uncomplete=0;
    ChannelExclude=[1  10 11];
    P_C=gBStrigger(P_C,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);
    
    %Load Class Information
    class_info=abs(textread(classinfo,'','delimiter',' '));
    name_classes={
        '10HZ'
        '11HZ'
        '12HZ'
        '13HZ'
    };
    use_rows=[1  2  3  4];
    P_C=gBSloadclass(P_C,class_info,name_classes,use_rows);
    
    % START: Minimum energy
    ChannelExclude = [];
    Frequencies = frequencies;       %[10  11  12  13];
    Replace = 'add channels';
    FileName = '';
    ProgressBarFlag = 0;
    P_C = gBSminimumenergy(P_C, ChannelExclude, Frequencies, NumberOfHarmonics,...
                       ModelOrder, IntervalLength, EvaluationStep,...
                       Replace, FileName, ProgressBarFlag);
    
    % Moving Average Filter
    TrialExclude = [];
    ChannelExclude=[1  2  3  4  5  6  7  8];
    Method = 'median';
    IntervalLength = 10;
    Replace = 'add channels';
    FileName = '';
    ProgressBarFlag = 0;
    P_C = gBSmovingwindowfilter(P_C, TrialExclude, ChannelExclude, Method , IntervalLength,...
        ProgressBarFlag);
    
    
    % Transform
    ApplyOn = 'multiple channels';
    ChannelExclude_mult = [1  2  3  4  5  6  7  8];
    TrialExclude_mult = [];
    Operation_mult = 'Z-MEDIAN';
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
                    
    % Visualize the new data
    %[V_R]=plot(P_C,V_R);
    
    %LDA Classifier: 10 x 10 cross validation               
    %----------------------------------------
    %Feature Matrix
    %Interval=[1   3  86];
    AttributeName={
        '10HZ'
        '11HZ'
        '12HZ'
        '13HZ'
    };
    ChannelExclude=[1 2 3 4 5 6 7 8];
    Permutate=0;
    MergeTimePoints=0;
    FileName=[''];
    ProgressBarFlag=[0];
    % TODO this works until gbsanalyze 3.10.00
    %F_O=gBSfeaturematrix(P_C,Interval,AttributeName,Permutate,ChannelExclude,FileName,ProgressBarFlag);
    
    % TODO this is required for gbsanlyze 4.11.00 and higher
    F_O=gBSfeaturematrix(P_C,Interval,AttributeName,Permutate,MergeTimePoints,ChannelExclude,FileName,ProgressBarFlag);
    
    %Linear Classifier
    PlotFeatures=[1  2];
    Method=['LDA'];
    P.metric=[''];
    TrainTestData=['CV'];
    
    FileName=[''];
    ProgressBarFlag=[0];
    C_O=gBSlinearclassifier(F_O,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);
    
    tmp=CreateResult2D(C_O);
    gResult2d(tmp)
    
    %LDA Classifier: Train 100%/Test 100%
    
    %Linear Classifier
    PlotFeatures=[1  2];
    Method=['LDA'];
    P.metric=[''];
    TrainTestData=['100:100'];
    FileName=[save_path];
    ProgressBarFlag=[0];
    C_O=gBSlinearclassifier(F_O,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);
catch err
    disp(err.message)
    P_C = old_PC;
end