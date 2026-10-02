%##########################################################################
% Accuracy for Online Classification
%--------------------------------------------------------------------------
% Author: Patrick Reitner
% Created At: 10.04.2012
% Last Modified At: 14.05.2012
% Filename: PhysioObserver_online_accuracy_batch_V0_1.m
%--------------------------------------------------------------------------
% This batch takes the loaded P_C data and calculates the accuracy of the
% CLASS channel which was measured during a feedback run.
%##########################################################################

global P_C
spath = mfilename('fullpath');
[BatchPath dn de] = fileparts(spath);
[ClassInfoPath dn de] = fileparts(BatchPath);

sampFrequency = P_C.SamplingFrequency;

%Classinfo File
%--------------
ClassInfoFile = 'CircleTraining-ClassInfo.m';
SelectClass = [ 2 3 4 ]; % MATH, SLEEP/REST, SPORT
MapClass = [ 1 3 ];

%Name of figure saved
%--------------
AccuracyPostFix = '_Accuracy';

%Trigger-settings
%------------------
TriggerLow = 2;
TriggerHigh = 57;
SamplesBefore = round(TriggerLow*sampFrequency); 
SamplesAfter = round(TriggerHigh*sampFrequency);

%Define Conditions
%0 -> all Trials
%1 -> discard first trial of exercise to ensure steady-state conditions
DiscardFirstTrial = 1;

old_PC = P_C;

try
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
    ChannelExclude=[1:size(P_C.Data,3)-1];
    P_C=gBStrigger(P_C,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);
    
    classResult = P_C.Data;
    
%     [ trials exercises ] = find(classfile);
%     trialexclude = [1; find(exercises(2:end) ~= exercises(1:end-1) ) + 1];
%     channelexclude=[];
%     p_c=gbscuttrialschannels(p_c,trialexclude,channelexclude);
    
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
    snr = mean( result(1,:) / result(2,:) );
    error_ = (1-result(1,:) ).*100;
    fp = result(2,:) .* 100;
    
    time = linspace(0,TriggerLow+TriggerHigh,size(error_,2));
    
    fig_handle = figure('Name','PhysioObserver Online Accuracy');
    plot(time,[ error_; fp])
    ylim([0 100]);
    line([3 3],[0 100],'LineWidth',4,'Color','r');
    legend('Error','False Positiv')
    xlabel(sprintf('time [s]: SNR = %.2f',snr));
    ylabel('Classification Error/False Positiv [%]')
    title('Online accuracy')
    % Check if user wants to save FeatureMatrix
    if ~isempty(AccuracyPostFix)
        % Set Accuracy- Filename
        [fpath fname fext] = fileparts(P_C.FileName);
        FileName=fullfile(fpath,[fname AccuracyPostFix]);
        % Save Figure in PNG- Format
        print(fig_handle, '-dpng', FileName);
    end
catch err
    % Show error message and restore data
    disp(getReport(err));
    P_C = old_PC;
end