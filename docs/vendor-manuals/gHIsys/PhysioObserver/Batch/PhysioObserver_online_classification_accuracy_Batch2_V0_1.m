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

% CLASSIFICATION SETTINGS
% Data for Classification
PC_Classify = PC_subj09_01; 

sampFrequency = PC_Classify.SamplingFrequency;

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
%FeatureSelect=[2,3,5,9,10,11,12,13,20,21,22]; %1st SET

%FeatureSelect=[1,4,14,16,17,18,20]; %empiric-feature selection SET
%FeatureSelect=[2,3,4,5,6,7,9,11,12,14,16,17,18,19,22,23,24,25,26,27,28,29,30,31,32,33,39];
%single-trial analysis SET

%FeatureSelect=[1,2,3,4,5,6,7,9,11,16,17,18,19,22,27,28,33]; % ANOVA feature selection SET
%FeatureSelect=[2,3,4,5,11,16,17,18,22,27]; % modified 1 ANOVA feature selection SET
%FeatureSelect=[2,4,6,7,11,16,17,18,22,27,28,33]; % modified 2 ANOVA feature selection SET -> e: mean 45% / fp: mean 0.01
%FeatureSelect=[2,4,6,7,11,16,17,18,22,27,28,33]; % modified 3 ANOVA feature selection SET
%FeatureSelect=[1,2,3,4,5,6,7,9,10,11,16,17,18,19,22,23,27,33]; % ANOVA merge feature selection SET
%FeatureSelect=[2,4,5,6,9,10,11,16,17,22,23,27,33];
%FeatureSelect=[2,4,6,7,10,11,16,17,18,22,27]; % modified 1 ANOVA merge feature selection SET

%FeatureSelect=[2,6,9,11,16,17,20,22,24,28,29,33,34]; % DSLVQ Feature Weighting feature selection (3 Parts > 0.1)
%FeatureSelect=[2,6,9,11,16,17,20,23,24,25,26,28,29,33,34,36,37,38,39]; %  DSLVQ Feature Weighting feature selection (weight sum > 0.9)
%FeatureSelect=[2,6,9,11,16,17,20,24,28,29,33,36,37,38];%DSLVQ Feature Weighting feature selection (weight > 1/39 (noise probability))
%FeatureSelect=[2,6,9,11,16,17,20,24,28,29,33]; %DSLVQ Feature Weighting feature selection (weight > 1/39 (noise probability))
%FeatureSelect=[2,6,9,11,16,17,20,22,23,24,28,29,33,34,35];

% FWC: unsorted, no change rates
%FeatureSelect=[2,6,7,9,12,16,28,29,31,32,34,35,37,38,41]; 

% 2. weighting of retrieved weight features
% close to singular or badly scaled
%FeatureSelect=[11,16,17,19,23,26,32,35,38,41,43,44,49,50,52,55,61,62];
%FeatureSelect=[2,11,16,17,19,23,26,32,35,38,41,44,50,52,55,61,62];

% FWC: unsorted, lt change rates -> matrix close to singular or badly
% scaled
%FeatureSelect=[2,6,7,9,12,16,28,29,31,32,34,35,37,38,41,44,47,49,50,55,56,58,59,62];


% CFWC: unsorted, no change rates
% matrix close to singular or badly scaled
%FeatureSelect = [2,6,7,9,11,12,18,19,20,23,24,25,26,27,28,29,31,32,34,35,37,38];
%FeatureSelect = [2,6,7,9,11,12,18,19,20,25,26,27]; % e: mean 6.09 / fp: 0.01

% FWC: group mixture + comparison
%FeatureSelect=[2,9,12,19,28,29,31,32,35,37,38,41];

% CFWC: group mixture + comparison
%FeatureSelect=[2,6,7,9,11,12,18,19,20,23,25,26,28,29,31,32,34,35,37,38];
%Warning: Matrix is close to singular or badly scaled.

%FeatureSelect=[2,6,7,9,11,12,18,19,20,23,29,31,32,34,35,37,38];

% original FWC feature weighting result
%FeatureSelect = [2,6,7,9,12,16,19,28,29,31,32,34,35,37,38,41];
% modified feature weighting result
FeatureSelect = [2,6,7,9,12,16,19,28,31,34,35,38,41];

% original CFWC feature weighting result
%FeatureSelect = [2,6,7,9,11,12,16,18,19,20,28,31,34,35,38,41];

% Define Output FileName -> '' = don't save
ClassifierPostFix='';
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
% Data for Classifier Accuracy
PC_Accuracy = PC_subj09_02;

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

P_C = PC_Classify;
old_PC1 = PC_Classify;
old_PC2 = PC_Accuracy;

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
        FileName=fullfile(fpath,[fname ClassifierPostFix fext]);
    else
        FileName='';
    end
    C_O=gBSlinearclassifier(F_M,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);
    
    if ShowResult2D == 1
        tmp=CreateResult2D(C_O);
        gResult2d(tmp);
    end

    % Restore data
    P_C = PC_Accuracy;

    % Load ClassInfo and Map Class 1 (Rest) to Class 3 (Sleep)
    classinfo = fullfile(ClassInfoPath,ClassInfoFile);
    classfile = load(classinfo);
    
    for map = 1:size(MapClass,1)
        classfile(logical(classfile(:,MapClass(map,1))),MapClass(map,2)) = 1;
    end
    
    classfile = classfile(:,SelectClass);
    class_ = zeros(1,size(classfile,1));
    
    for cntTrial=1:size(classfile,1)
        nextclass = find(classfile(cntTrial,:),1,'first');
        if isempty(nextclass)
            nextclass = 0;
        end
        class_(cntTrial) = nextclass;
    end

    %Trigger
    New_tm{1}={size(P_C.Data,3)-1, 1, 'v', 0.9, 0, 'TRIG', 'red'};
    %SamplesBefore=1024;
    %SamplesAfter=3328;
    Uncomplete=0;
    ChannelExclude=[1:size(P_C.Data,3)];
    %Unselect selected feature channels
    ChannelExclude(FeatureSelect+1) = [];
    P_C=gBStrigger(P_C,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);

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
        
        %cut trials from Class- Compare Structure 
        class_(trials) = [];
    end
    
    P_C_Trig=P_C;
    
    %Load Classifier
    if ~strcmp('',Classifier)
        % Classifier defined -> use defined classifier
        C_O_S=classifierobj;
        FileName=Classifier;
        C_O_S=load(C_O_S,FileName);
    else
        % No classifier defined -> use calculated classifier
        C_O_S=C_O;
    end
    
    wb=waitbar(0,'Please wait while carrying out operations');
    bestFP.correct = 1;
    bestE.e = 100;
    for cp=ClassifierPoints
        waitbar((cp-min(ClassifierPoints))/(max(ClassifierPoints)-min(ClassifierPoints)),wb,'Please wait, while carrying out operations!');
        P_C=P_C_Trig;
        %Apply Classifier
        ClassifierNumber=cp;
        ShowCombined=1;
        Replace='replace all channels';
        FileName='';
        ProgressBarFlag=0;
        ConfidenceInterval=ZeroClassConfidence;
        OutputProbabilities=0;
        
        %fprintf(1,'before apply classifier');
        %fprintf(2,'before apply classifier');
        P_C=gBSapplyclassifier(P_C,C_O_S,ClassifierNumber,Replace,ShowCombined,ConfidenceInterval,OutputProbabilities,FileName,ProgressBarFlag);
        %fprintf(1,'after apply classifier');
        %fprintf(2,'after apply classifier');
        
        classResult = P_C.Data;
        
        %compare Class.Result with classinfo
        result = zeros(2,size(classResult,2));
        for cntTrial=1:size(classResult,1)
            for cntSamp=1:size(classResult,2)
                if  classResult(cntTrial,cntSamp) == class_(1,cntTrial)
                    result(1,cntSamp) = result(1,cntSamp) + 1;
                elseif classResult(cntTrial,cntSamp) ~= 0
                    result(2,cntSamp) = result(2,cntSamp) + 1;
                end
            end
        end
        result = result./(size(classResult,1));
        correct = mean(result(2,:));
        if  max(result(2,:)) ~= 0 
            snr = result(1,:) ./ result(2,:);
            snr = mean(snr(~isnan(snr)));
        else
            snr = 0;
        end
        error_ = (1-result(1,:) ).*100;
        e = mean(error_);
        fp = result(2,:) .* 100;
        % Compare calculated result with best result (False-Positive best)
        if (correct < bestFP.correct)
            % Save result as best result
            bestFP.correct = correct;
            bestFP.cp = cp;
            bestFP.snr = snr;
            bestFP.error = error_;
            bestFP.fp = fp;
        end
        
        % Compare calculated result with best result (error best)
        if (e < bestE.e)
            % Save result as best result
            bestE.e = e;
            bestE.cp = cp;
            bestE.snr = snr;
            bestE.error = error_;
            bestE.fp = fp;
        end
        
        
        
        
        % Show all timepoints?
        if (ShowAccuracyPlot == 1)
            time = linspace(0,TriggerLow+TriggerHigh,size(error_,2));
            
            fig_handle = figure('Name','PhysioObserver Online Classification And Accuracy');
            plot(time,[ error_; fp])
            ylim([0 100]);
            line([3 3],[0 100],'LineWidth',4,'Color','r');
            legend('Error','False Positiv')
            xlabel(sprintf('time [s]:   SNR = %.2f   ClassifierPoint = %d     Mean = %.2f',snr,cp,correct));
            ylabel('Classification Error/False Positiv [%]')
            title('Online classification and accuracy')
        end
    end
    close(wb);
    
    time = linspace(0,TriggerLow+TriggerHigh,size(error_,2));
    % Show best timepoint
    fig_handle1 = figure('Name','PhysioObserver Online Classification And Accuracy');
    plot(time,[ bestFP.error; bestFP.fp]);
    ylim([0 100]);
    line([3 3],[0 100],'LineWidth',4,'Color','r');
    legend('Error','False Positiv');
    xlabel(sprintf('time [s]: SNR = %.2f ClassifierPoint = %d     Mean = %.2f',bestFP.snr,bestFP.cp,bestFP.correct));
    ylabel('Classification Error/False Positiv [%]');
    title('Online classification and accuracy');

    % Show best timepoint
    fig_handle2 = figure('Name','PhysioObserver Online Classification And Accuracy');
    plot(time,[ bestE.error; bestE.fp]);
    ylim([0 100]);
    line([3 3],[0 100],'LineWidth',4,'Color','r');
    legend('Error','False Positiv');
    xlabel(sprintf('time [s]: SNR = %.2f ClassifierPoint = %d     Mean = %.2f',bestE.snr,bestE.cp,bestE.e));
    ylabel('Classification Error/False Positiv [%]');
    title('Online classification and accuracy');

    % Check if user wants to save FeatureMatrix
    if ~isempty(AccuracyPostFix)
        % Set Accuracy- Filename
        [fpath fname fext] = fileparts(P_C.FileName);
        FileName=fullfile(fpath,[fname AccuracyPostFix]);
        % Save Figure in PNG- Format
        print(fig_handle, '-dpng', FileName);      
    end
    
catch err
    disp(getReport(err));
    PC_Classify = old_PC1;
    PC_Accuracy = old_PC2;
end