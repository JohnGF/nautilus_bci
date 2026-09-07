% P300 Speller Batch for gBSanalyze version 3.0
% g.tec medical engineering GmbH

%Use this batch for the data stored with gMOBIlabP300SCF.mdl or gUSBP300SCF.mdl

global P_C
global V_R
[pathstr]=fileparts(P_C.FileName);

%check if data contains 4 channels -> eliminate time stamp
y=P_C.Data;
if size(y,3)==4
    TrialExclude=[];
    ChannelExclude=[1];
    P_C=gBScuttrialschannels(P_C,TrialExclude,ChannelExclude);
end
stack_P_C=P_C;


%----------------------------------------
%Step 1 - Find the flash time points
% Eventfinder (overflow)
MarkOverflow = 1; showEpochingAreas_over = 0; setStartMarker_over = 1; setStopMarker_over = 0;
AssignAttribute_over = 0; StartMarker_over = 'S'; StopMarker_over = 'OR2';
TrialAttribute_over = 'OVERRUN'; Threshold_over = 0.1; getUnit_over = 'µV';
TrialExclude_over = []; ChannelExclude_over = [1 3]; ProgressBarFlag = 0;
[P_C, PreviewOverflow, VecThreshold] = gBSoverflow...
        (P_C, MarkOverflow, showEpochingAreas_over,...
        setStartMarker_over, setStopMarker_over,...
        AssignAttribute_over, StartMarker_over, StopMarker_over,...
        TrialAttribute_over, Threshold_over, getUnit_over,...
        TrialExclude_over, ChannelExclude_over, ProgressBarFlag);

%Trigger
New_tm{1}={3 1 'NON' 'red'}; SamplesBefore=26; SamplesAfter=179; Uncomplete=0;
ChannelExclude=[3];
P_C=gBStrigger(P_C,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);

%Save
P_C_S=P_C;
save([pathstr,'\TN.mat'],'P_C_S');

%----------------------------------------
%Step 2 - Extract only targets
P_C=stack_P_C;

% Eventfinder (overflow)
MarkOverflow = 1; showEpochingAreas_over = 0; setStartMarker_over = 1;
setStopMarker_over = 0; AssignAttribute_over = 0; StartMarker_over = 'OR1';
StopMarker_over = 'T'; TrialAttribute_over = 'OVERRUN'; Threshold_over = 90;
getUnit_over = '% of max'; TrialExclude_over = []; ChannelExclude_over = [1  2];
ProgressBarFlag = 0;
[P_C, PreviewOverflow, VecThreshold] = gBSoverflow...
        (P_C, MarkOverflow, showEpochingAreas_over,...
        setStartMarker_over, setStopMarker_over,...
        AssignAttribute_over, StartMarker_over, StopMarker_over,...
        TrialAttribute_over, Threshold_over, getUnit_over,...
        TrialExclude_over, ChannelExclude_over, ProgressBarFlag);

%Trigger
New_tm{1}={3 1 'NON' 'red'}; SamplesBefore=26; SamplesAfter=179;
Uncomplete=0; ChannelExclude=[2];
P_C=gBStrigger(P_C,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);
P_C_T=P_C;
P_C_S=P_C;
save([pathstr,'\T.mat'],'P_C_S');

%----------------------------------------
%Step 3 - take the target and invert it, then add to TN ->TNMINUST
%all have class label NON -> needed for EP calculation

P_C=P_C_T;

% Transform
ApplyOn = 'multiple channels'; ChannelExclude_mult = [2]; TrialExclude_mult = [];
Operation_mult = 'MULT'; SecondOperand_mult(1) = -1; Unit_mult = 'µV';
FirstOperand_two = 1; Operation_two = 'SUB'; SecondOperand_two = [1];
ProgressBarFlag = 0;
P_C = gBSarithmetic(P_C, ApplyOn, ChannelExclude_mult,...
      TrialExclude_mult, Operation_mult, SecondOperand_mult,...
      Unit_mult, FirstOperand_two, Operation_two,...
      SecondOperand_two, ProgressBarFlag);

%Merge
FileName={[pathstr,'\TN.mat']}; Concatenate='Trials';
AdoptChAttr=1; AdoptTrialAttr=1; AdoptMarkers=1;
P_C=gBSmerge(P_C,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);
%Save
P_C_S=P_C;
save([pathstr,'\TNMINUST.mat'],'P_C_S');

%----------------------------------------
%Step 4 - assign the trial attribute Target to all target trials
P_C=P_C_T;
%P_C=load(P_C,'T.mat');
tmp=P_C.AttributeName;
tmp{3}='Target';
P_C.AttributeName=tmp;
P_C_TARGET=P_C;
P_C_S=P_C;
save([pathstr,'\Target.mat'],'P_C_S');

%----------------------------------------
%Step 5 - merge the Targets and NON Trials, 
%class lables: NON and TARGET
FileName={[pathstr,'\TNMINUST.mat']};
Concatenate='Trials'; AdoptChAttr=1; AdoptTrialAttr=1; AdoptMarkers=1;
P_C=gBSmerge(P_C,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);
P_C_ALL=P_C;    %contains all trials with Target and NON attribute
P_C_S=P_C;
save([pathstr,'\ALL.mat'],'P_C_S');

%----------------------------------------
%Step 6 - EP of Target trials, id 3
trial_id=[3]; channel_id=[]; type_id=[]; channelnr_id=[1]; flag_tr='tr_inc';
flag_ch='ch_exc'; flag_type='type_exc'; flag_nr='nr_inc';
[TrialExclude, ChannelExclude]=gBSselect(P_C,trial_id,flag_tr,channel_id,flag_ch,type_id,flag_type,channelnr_id,flag_nr);

%Average
Baseline=[1  26]; Smoothing={'none'}; DownSampling=0; FileName=''; Averaging='simple';
var1=0; var2=0; var3=0;
A_O = gBSaverage(P_C,Baseline,Smoothing,DownSampling,TrialExclude,ChannelExclude,FileName,Averaging,0,var1,var2,var3);
tmp=CreateResult2D(A_O); gResult2d(tmp);

%----------------------------------------
%Step 7 - EP of NON trials , id 4
trial_id=[4]; channel_id=[]; type_id=[]; channelnr_id=[1]; flag_tr='tr_inc';
flag_ch='ch_exc'; flag_type='type_exc'; flag_nr='nr_inc';
[TrialExclude, ChannelExclude]=gBSselect(P_C,trial_id,flag_tr,channel_id,flag_ch,type_id,flag_type,channelnr_id,flag_nr);

%Average
Baseline=[1  26]; Smoothing={'none'}; DownSampling=0; FileName=''; Averaging='simple';
var1=0; var2=0; var3=0;
A_O = gBSaverage(P_C,Baseline,Smoothing,DownSampling,TrialExclude,ChannelExclude,FileName,Averaging,0,var1,var2,var3);
tmp=CreateResult2D(A_O); gResult2d(tmp);

