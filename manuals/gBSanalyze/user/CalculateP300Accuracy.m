% example code for using the P300 Accuracy function in gBSanalyze
% g.tec medical engineering GmbH
% Created by Rupert Ortner 2012, last update April 2015
%
% This batch contains two parts:
%        part 1: creating a classifier
%        part 2: applying the classifier onto the data

%% get mode (create a classifier, apply a classifier, do both)
handle = figure('Name','Mode','Units','pixels','visible','off','MenuBar','none','IntegerHandle','off','NumberTitle','off');
set(gcf,'Position',[1000,500,240,120],'visible','on');
aa = uicontrol(handle, 'Style', 'PopupMenu', ...
    'Position', [20 60 200 40], ...
    'String', {'Calculate and Apply a Classifier','Calculate a Classifier', 'Apply a Classifier'});
sb = uicontrol('Position',[20 20 200 40],'String','OK !','BackgroundColor',[ 0.702, 0.702, 0.702],'FontSize',10,...
    'FontWeight','bold','Callback','uiresume(gcbf)');
uiwait(gcf);
ControlFlag = get(aa,'Value');
close(handle);

global P_C;
smpf = P_C.SamplingFrequency;

%% part 1: create a classifier
if (ControlFlag == 1) || (ControlFlag == 2)
    wb = waitbar(0,'Please wait, while carrying out operations!');
    set(wb, 'name', 'Calculating P300 Classifier ...');
    try
        % part 1: create a classifier
        P_C_copy1 = P_C;
        %% if the first channel contains the timestamp => remove it
        dat_temp1 = P_C_copy1.Data;
        firstchannel = dat_temp1(:,:,1);
        ascent = diff(firstchannel);
        
        if  length((unique(round(ascent*10^6)))==1) ;   % if the diff of first channel is constant, rounded after 10^6 => this channels contains the timestamp and will be cutted
            TrialExclude=[];
            ChannelExclude=1;
            P_C_copy1 = gBScuttrialschannels(P_C_copy1,TrialExclude,ChannelExclude);
        end
        clear dat_temp1 firstchannel;
        waitbar(1/9);
        %%
        dat_temp2 = P_C_copy1.Data;
        [NTrials,NSamples,NChannels] = size(dat_temp2);
        triggerData = dat_temp2(:,:,NChannels - 1);
        targetData = dat_temp2(:,:,NChannels);
        triggerTime = find(diff(triggerData) > 0);
        targetTime = find(diff(targetData) > 0);
        classInfo = zeros(2,length(triggerTime));
        for NT = 1:length(triggerTime)
            if isempty(intersect(triggerTime(NT),targetTime))
                classInfo(:,NT) = [1;0];
            else
                classInfo(:,NT) = [0;1];
            end
        end
        clear dat_temp2 triggerData targetData;
        waitbar(2/9);
        %% Bandpass filter 0.1 Hz to 30 Hz
        clear Filter;
        Filter.Realization = 'butter';
        Filter.Type = 'BP';
        Filter.Order = 4;
        Filter.f_high = 30;
        Filter.f_low = 0.1;
        FiltFiltFlag = 1;
        TrialExclude = [];
        ChannelExclude = [NChannels-1  NChannels];
        P_C_copy1 = gBSfilter(P_C_copy1,Filter,FiltFiltFlag,ChannelExclude,TrialExclude);
        waitbar(3/9);
        %%  Moving Window Filter
        TrialExclude = [];
        Method = 'average';
        IntervalLength = 12;
        ProgressBarFlag = 0;
        P_C_copy1 = gBSmovingwindowfilter(P_C_copy1, TrialExclude, ChannelExclude, Method,IntervalLength, ProgressBarFlag);
        waitbar(4/9);
        %% Trigger
        New_tm{1}={NChannels-1  1 'v' 0.01 1};
%         SamplesBefore=26;
%         SamplesAfter=179;
        SamplesBefore = floor(0.1016*smpf);
        SamplesAfter=(round(0.6992*smpf));
        Uncomplete=0;
        P_C_copy1=gBStrigger(P_C_copy1,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);
        waitbar(5/9);
        %% load ClassInfo
        name_classes={
            'NONTARGET'
            'TARGET'
            };
        use_rows=[1  2];
        P_C_copy1=gBSloadclass(P_C_copy1,classInfo,name_classes,use_rows);
        waitbar(6/9);
        %% Baseline Correction
%         Interval = [1  26];
        Interval = [1 floor(0.1016*smpf)];
        ChannelExclude = 1;
        TrialExclude = [];
        ProgressBarFlag = 0;
        P_C_copy1 = gBSbaselinecorrection(P_C_copy1, Interval, ChannelExclude,TrialExclude, ProgressBarFlag);
        waitbar(7/9);
        %% create the Feature Matrix
        %Interval=[27 12 205];
        Interval=[floor(0.1055*smpf) 12 floor(0.8008*smpf)];
        AttributeName={
            'NONTARGET'
            'TARGET'
            };
        ChannelExclude=[];
        Permutate=0;
        MergeTimePoints=1;
        FileName='';
        ProgressBarFlag = 0;
        F_M = gBSfeaturematrix(P_C_copy1,Interval,AttributeName,Permutate,MergeTimePoints,ChannelExclude,FileName,ProgressBarFlag);
        waitbar(8/9);
        %% create the Linear Classifier
        PlotFeatures=[1  2];
        Method='LDA';
        P.metric='';
        TrainTestData='100:0';
        FileName='P300classifier.mat';
        ProgressBarFlag = 0;
        C_O=gBSlinearclassifier(F_M,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);
        clear P_C_copy1;
        waitbar(9/9);
        close(wb);
        disp(['The classifier was saved as: ',FileName]);
        clear P_C_copy1;
    catch
        close(wb);
        troubles('An error occured while calculating the P300 Classifier','Please verify that the loaded data is a valid P300 data file');
    end
end
%% -------------------------------------------------------------------------------------------
%---------------------------------------------------------------------------------------------
% part 2: apply the classifier onto the data
%---------------------------------------------------------------------------------------------
%---------------------------------------------------------------------------------------------
if (ControlFlag == 1) || (ControlFlag == 3)
    wb = waitbar(0,'Please wait, while carrying out operations!');
    set(wb, 'name', 'Apply the P300 Classifier ...');
    try
        %% get the Classifier
        if ControlFlag == 1
            Classifier = C_O;
        elseif ControlFlag == 3
            [filename, pathname, ~] = uigetfile( ...
                {'*.mat','MAT-files (*.mat)'}, ...
                'Pick a file');
            Classifier = classifierobj;
            Classifier = load(Classifier,[pathname,filesep,filename]);
        end
        P_C_copy2 = P_C;        
        %% if the first channel contains the timestamp => remove it
        dat_temp1 = P_C_copy2.Data;
        firstchannel = dat_temp1(:,:,1);
        ascent = diff(firstchannel);
        
        if  length((unique(round(ascent*10^6)))==1) ;   % if the diff of first channel is constant, rounded after 10^6 => this channels contains the timestamp and will be cutted          
            ChannelExclude=1;
            TrialExclude = [];
            P_C_copy2 = gBScuttrialschannels(P_C_copy2,TrialExclude,ChannelExclude);
        end
        clear dat_temp1 firstchannel;
        waitbar(1/3);
        NChannels = size(P_C_copy2.Data,3);        
        %% Bandpass filter 0.1 Hz to 30 Hz
        clear Filter;
        Filter.Realization = 'butter';
        Filter.Type = 'BP';
        Filter.Order = 4;
        Filter.f_high = 30;
        Filter.f_low = 0.1;
        FiltFiltFlag = 1;
        TrialExclude = [];
        ChannelExclude = [NChannels-1  NChannels];
        P_C_copy2 = gBSfilter(P_C_copy2,Filter,FiltFiltFlag,ChannelExclude,TrialExclude);
        waitbar(2/3);        
        %% calculate the P300 accuracy
        FileName='';
        ProgressBarFlag=0;
        P_O = gBSP300Accuracy(P_C_copy2, Classifier, FileName, ProgressBarFlag);
        R_O = CreateResult2D(P_O);
        waitbar(3/3);
        close(wb);
        gResult2d(R_O);
        clear P_C_copy2;
    catch
        close(wb);
        troubles('An error occured while applying the P300 Classifier','Please verify that the loaded data is a valid P300 data file');
    end
end