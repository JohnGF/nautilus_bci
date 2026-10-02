%##########################################################################
%  Accuracy for Offline Classification
%--------------------------------------------------------------------------
% Author: Patrick Reitner
% Created At: 10.04.2012
% Last Modified At: 19.06.2012
% Filename: PhysioObserver_online_classification_accuracy_Batch2_V0_1.m
%--------------------------------------------------------------------------
% This batch takes the configured P_C data and calculates a classifier. 
% The accuracy is calculated by applying the calculated or a configured 
% classifier on the specified feedback- P_C data.
%##########################################################################


global P_C

spath = mfilename('fullpath');
[BatchPath dn de] = fileparts(spath);
[ClassInfoPath dn de] = fileparts(BatchPath);
[ DataPath DataName DataExt] = fileparts(P_C.FileName);
if isempty(DataPath)
    DataPath = pwd;
end
if isempty(DataName)
    DataName = 'Session1';
end
save_path = fullfile( DataPath , sprintf('%s_train_lda.mat',DataName));
% CLASSIFICATION SETTINGS
% Data for Classification

sampFrequency = P_C.SamplingFrequency;

% Set Classification Intervall
ClassificationStep = 1; % [s]
ClassificationEnd = 59; % [s]
PreTrigger = 2; %[s]
PostTrigger = 57; %[s]

% Define Class- Names
ClassNames={
    'REST'
    'D2'
    'SLEEP'
    'SPORT'
    };

% Show Result2D -> 1 = show, 0 = don't show
ShowResult2D = 1;
% Define Feature- Matrix- Classes
% 1...REST, 2...MATH, 3...SLEEP, 4...SPORT
SelectClasses = [1,2,3,4];

% Define Feature- Channels for Classifier- Calculation
FeatureSelect = [2,6,7,9,12,16,19,28,31,34,35,38,41];


% Define Output FileName -> '' = don't save
ClassifierPostFix='train_lda';
FeaturePostFix='';

% TrainTestDataRation- Options:
% Train:Test [%]:[%]
% '50:50'
% '100:100'
% '100:0'
% for 10 x 10 Cross- Validation use 'CV'
TrainTestDataRatio='100:100';

% Classinfo File
ClassInfoFile = 'CircleTraining-ClassInfo.m';
SelectClass = SelectClasses;%[ 2 3 4 ]; % MATH, SLEEP/REST, SPORT
%MapClass = [ 1 3 ];
MapClass = [ ];

% Define Classification- Method
ClassificationMethod='LDA';

% CLASSIFIER SETTINGS


% Classifier- Object -> '' = use calculated Classifier
Classifier = '';

% Classifier- Points
% one ClassifierPoint x -> [x:x]
% range of ClassifierPoint x to ClassifierPoint y -> [x:y]
% all ClassifierPoints -> [1:59]
ClassifierPoints = [56:59];

% ACCURACY SETTINGS
% Zero- Class- Confidence Interval
ZeroClassConfidence = 5;

% Accuracy- Plot- Setting
% 1...Show all Timepoints, 2...Show best Timepoint
ShowAccuracyPlot = 2 ;

% Name of figure saved -> '' = don't save
% When a range of classifierPoints is configured, the figure of the best ClassifierPoint
% will be saved
AccuracyPostFix = '';

% TRIGGER SETTINGS
TriggerLow = 2;
TriggerHigh = 57;
SamplesBefore = round(TriggerLow*sampFrequency); 
SamplesAfter = round(TriggerHigh*sampFrequency);

% CONDITION SETTINGS
DiscardFirstTrial = 1;


old_PC = P_C;


try
    %Select Trials and Channels
    trial_id=[];
    channel_id=[];
    type_id=[];
    %cut out Time (CH01), Trigger (pre-last channel) and  Class- Result (last Channel)
    channelnr_id=[1,size(P_C.Data,3)-1,size(P_C.Data,3)];
    flag_tr='tr_exc';
    flag_ch='ch_exc';
    flag_type='type_exc';
    flag_nr='nr_exc';
    [TrialExclude, ChannelExclude]=gBSselect(P_C,trial_id,flag_tr,channel_id,flag_ch,type_id,flag_type,channelnr_id,flag_nr);
    
    %Trigger
    %Trigger on  Trigger Input (pre-last Channel), rising edge, threshold level 90%, accept
    %incomplete, accept overlap, name -> 'EX', color -> red
    New_tm{1}={size(P_C.Data,3)-1, 1 ,'l', 90, 1, 'EX', 'red'};
    SamplesBefore=round(PreTrigger * P_C.SamplingFrequency);
    SamplesAfter=round(PostTrigger * P_C.SamplingFrequency);
    Uncomplete=1;
    ChannelExclude=[ 1, size(P_C.Data,3) - 1,size(P_C.Data,3) ];
    P_C=gBStrigger(P_C,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);
    
    %Load Class Information
    class_info=textread(fullfile(ClassInfoPath,ClassInfoFile),'','delimiter',' ');
    name_classes = ClassNames;
    use_rows=1:length(name_classes);
    P_C=gBSloadclass(P_C,class_info,name_classes,use_rows);
    
    %Discard first trial of each exercise to ensure steady- state
    %conditions
    if DiscardFirstTrial == 1
        %this snippet only works for non- merged class info files
        %[ trials exercises ] = find(class_info);
        %TrialExclude = [1; find(exercises(2:end) ~= exercises(1:end-1) ) + 1];
        
        %this snippet only works for merged class info files too
        [trials exercises] = find(class_info(1:end-1,:) ~= class_info(2:end,:));
        TrialExclude = [1, transpose(unique(trials)) + 1];
        
        ChannelExclude=[];
        P_C=gBScuttrialschannels(P_C,TrialExclude,ChannelExclude);
    end
    
    %Feature Matrix
    Interval = [1, ClassificationStep*P_C.SamplingFrequency, ClassificationEnd*P_C.SamplingFrequency];
    AttributeName=ClassNames(SelectClasses);
    ChannelExclude=1:size(P_C.Data,3);
    ChannelExclude(FeatureSelect) = [];
    Permutate=0;
    MergeTimePoints=0;
    % Check if user wants to save FeatureMatrix
    if ~isempty(FeaturePostFix)
        % Set FM- Filename
        [ fpath fname fext] = fileparts(P_C.FileName);
        FileName=fullfile(fpath,[fname FeaturePostFix fext]);
    else
        FileName='';
    end
    ProgressBarFlag=0;
    F_M=gBSfeaturematrix(P_C,Interval,AttributeName,Permutate,MergeTimePoints,ChannelExclude,FileName,ProgressBarFlag);
    
    %Linear Classifier
    PlotFeatures=[1  2];
    Method=ClassificationMethod;
    P.metric='';
    TrainTestData=TrainTestDataRatio;
    ProgressBarFlag=0;
    % Check if user wants to save ClassifierObject
    if ~isempty(ClassifierPostFix)
        % Set CO- Filename
        [fpath fname fext] = fileparts(P_C.FileName);
        FileName=fullfile(fpath,sprintf('%s_%s%s',fname,ClassifierPostFix,fext));
    else
        FileName='';
    end
    C_O=gBSlinearclassifier(F_M,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);
    
    if ShowResult2D == 1
        tmp=CreateResult2D(C_O);
        gResult2d(tmp);
    end
    %LDA Classifier: Train 100%/Test 100%
    
   
catch err
    disp(getReport(err));
    P_C = old_PC;
  
end