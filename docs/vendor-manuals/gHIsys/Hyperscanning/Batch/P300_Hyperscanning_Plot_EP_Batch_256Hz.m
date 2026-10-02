% P300 Hyperscanning Speller Batch for gBSanalyze version 5.0
% g.tec medical engineering GmbH

% Use this batch for the data stored with gHyperscanning_ P300 models 
% run on postfixes: P300

global P_C
global V_R
[pathstr]=fileparts(P_C.FileName);

dataSampFreq = P_C.SamplingFrequency;
ratio_before = 0.1; SamplesBefore = ceil(ratio_before * dataSampFreq);
ratio_after = 0.65; SamplesAfter = ceil(ratio_after * dataSampFreq);

% Determine the number of subjects
if size (P_C.Data,3) == 19
    SubjNr = 2;
elseif size(P_C.Data,3) == 35
    SubjNr = 4;
elseif size(P_C.Data,3) == 67
    SubjNr = 8;
elseif size(P_C.Data,3) == 11
% elseif size(P_C.Data,3) == 10
    SubjNr = 1;
end

for i=1:SubjNr
    P_C_S = P_C;
    %Select Trials and channels
    if i == 1
        SubjFirstCh = 2;
    else
        SubjFirstCh = SubjFirstCh+8;
    end
    trial_id = [];
    channel_id=[];
    type_id=[];
    channelnr_id=[SubjFirstCh SubjFirstCh+1 SubjFirstCh+2 ...
        SubjFirstCh+3 SubjFirstCh+4 SubjFirstCh+5 SubjFirstCh+6 SubjFirstCh+7];
    flag_tr='tr_exc';
    flag_ch='ch_exc';
    flag_type='type_exc';
    flag_nr='nr_inc';
    [TrialExclude, ChannelExclude]=gBSselect(P_C_S,trial_id,flag_tr,channel_id,flag_ch,type_id,flag_type,channelnr_id,flag_nr);
    
    %Filter
    Filter.Realization='butter';
    Filter.Type='BP';
    Filter.Order=4;
    Filter.f_high=20;
    Filter.f_low=0.05;
    FiltFiltFlag = 0;
    TrialExclude=[];
    
    NrOfCh = size(P_C_S.Data,3);
    All_ch = 1:1:NrOfCh;
    
    ChannelExclude=setdiff(All_ch,channelnr_id);
    P_C_S=gBSfilter(P_C_S,Filter,FiltFiltFlag,ChannelExclude,TrialExclude);

    % eliminate timestamp and the other users channels
    y=P_C_S.Data;
    if size(y,3)==11 || size(y,3)==19 || size(y,3)==35 || size(y,3)==67
        TrialExclude=[];
        SubjIch=horzcat(channelnr_id,NrOfCh-1,NrOfCh);       
        ChannelExclude=setdiff(All_ch,SubjIch);
        P_C_S=gBScuttrialschannels(P_C_S,TrialExclude,ChannelExclude);
    end
        
    stack_P_C_S=P_C_S;
    
    %Step 1 - Find the flash time points
    % Eventfinder (overflow)
    MarkOverflow = 1; showEpochingAreas_over = 0; setStartMarker_over = 1; setStopMarker_over = 0;
    AssignAttribute_over = 0; StartMarker_over = 'S'; StopMarker_over = 'OR2';
    TrialAttribute_over = 'OVERRUN'; Threshold_over = 0.1; getUnit_over = 'µV';
    TrialExclude_over = []; 
    NrOfCh = size(P_C_S.Data,3);
    ChannelExclude_over = [1 2 3 4 5 6 7 8 10]; 
    ProgressBarFlag = 0;
    [P_C_S, PreviewOverflow, VecThreshold] = gBSoverflow...
        (P_C_S, MarkOverflow, showEpochingAreas_over,...
        setStartMarker_over, setStopMarker_over,...
        AssignAttribute_over, StartMarker_over, StopMarker_over,...
        TrialAttribute_over, Threshold_over, getUnit_over,...
        TrialExclude_over, ChannelExclude_over, ProgressBarFlag);
    
    %Trigger
    New_tm{1}={3 1 'NON' 'red'}; Uncomplete=0;
    ChannelExclude=[10];
    P_C_S=gBStrigger(P_C_S,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);
    
    %Save
    save([pathstr,'\TN.mat'],'P_C_S');
    
    %----------------------------------------
    %Step 2 - Extract only targets
    P_C_S=stack_P_C_S;
    
    % Eventfinder (overflow)
    MarkOverflow = 1; showEpochingAreas_over = 0; setStartMarker_over = 1;
    setStopMarker_over = 0; AssignAttribute_over = 0; StartMarker_over = 'OR1';
    StopMarker_over = 'T'; TrialAttribute_over = 'OVERRUN'; Threshold_over = 90;
    getUnit_over = '% of max'; TrialExclude_over = []; 
    ChannelExclude_over = [1 2 3 4 5 6 7 8 9];
    ProgressBarFlag = 0;
    [P_C_S, PreviewOverflow, VecThreshold] = gBSoverflow...
        (P_C_S, MarkOverflow, showEpochingAreas_over,...
        setStartMarker_over, setStopMarker_over,...
        AssignAttribute_over, StartMarker_over, StopMarker_over,...
        TrialAttribute_over, Threshold_over, getUnit_over,...
        TrialExclude_over, ChannelExclude_over, ProgressBarFlag);
    
    %Trigger
    New_tm{1}={3 1 'NON' 'red'}; 
    Uncomplete=0; ChannelExclude=[9];
    P_C_S=gBStrigger(P_C_S,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);
    P_C_T=P_C_S;
    save([pathstr,'\T.mat'],'P_C_S');
    
    %----------------------------------------
    %Step 3 - take the target and invert it, then add to TN ->TNMINUST
    %all have class label NON -> needed for EP calculation
    
    P_C_S=P_C_T;
    
    % Transform
    ApplyOn = 'multiple channels'; ChannelExclude_mult = [9]; TrialExclude_mult = [];
    Operation_mult = 'MULT'; SecondOperand_mult(1) = -1; Unit_mult = 'µV';
    FirstOperand_two = 1; Operation_two = 'SUB'; SecondOperand_two = [1];
    ProgressBarFlag = 0;
    P_C_S = gBSarithmetic(P_C_S, ApplyOn, ChannelExclude_mult,...
        TrialExclude_mult, Operation_mult, SecondOperand_mult,...
        Unit_mult, FirstOperand_two, Operation_two,...
        SecondOperand_two, ProgressBarFlag);
    
    %Merge
    FileName={[pathstr,'\TN.mat']}; Concatenate='Trials';
    AdoptChAttr=1; AdoptTrialAttr=1; AdoptMarkers=1;
    P_C_S=gBSmerge(P_C_S,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);
    %Save
    save([pathstr,'\TNMINUST.mat'],'P_C_S');
    
    %----------------------------------------
    %Step 4 - assign the trial attribute Target to all target trials
    P_C_S=P_C_T;
    %P_C=load(P_C,'T.mat');
    tmp=P_C_S.AttributeName;
    tmp{3}='Target';
    P_C_S.AttributeName=tmp;
    P_C_TARGET=P_C_S;
    save([pathstr,'\Target.mat'],'P_C_S');
    
    %----------------------------------------
    %Step 5 - merge the Targets and NON Trials,
    %class lables: NON and TARGET
    FileName={[pathstr,'\TNMINUST.mat']};
    Concatenate='Trials'; AdoptChAttr=1; AdoptTrialAttr=1; AdoptMarkers=1;
    P_C_S=gBSmerge(P_C_S,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);
    P_C_ALL=P_C_S;    %contains all trials with Target and NON attribute
    save([pathstr,'\ALL.mat'],'P_C_S');
    
    
    Baseline=[1 26];
    Smoothing={'average' 4};
    DownSampling=0;
    TrialExclude=[];
    ChannelExclude=[9];
    FileName='';
    Averaging='different';
    ClassIndex = [3 4];
    Threshold = [];
    var3=0;
    A_O = gBSaverage(P_C_S,Baseline,Smoothing,DownSampling, TrialExclude,...
        ChannelExclude,FileName,Averaging,0,ClassIndex,Threshold,var3);
    r2d = CreateResult2D(A_O);
    gResult2d(r2d);
    set(gcf,'Name',['Subject ' num2str(i)]);
end
