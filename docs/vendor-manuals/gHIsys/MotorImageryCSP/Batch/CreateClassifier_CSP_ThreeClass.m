% run on postfixes: MI
% clear;
%% settings
% evalwindow = [                      %evaluate the CSP in these windows
%     4.0 6.0;...
%     4.5 6.5;...
%     5.0 7.0;...
%     5.5 7.5;...
%     6.0 8.0];
evalwindow = [5 7];                 
detectionThreshold = 10;            % threshold for artifact detection
VarianceWindowLength = 1.5;         % length of window for calculation of variance (set the same value in the simulink model)
%%
global P_C;
%% enable or disable automatic artifact detection(create a classifier, apply a classifier, do both)
handle = figure('Name','Mode','Units','pixels','visible','off','MenuBar','none','IntegerHandle','off','NumberTitle','off');
set(gcf,'Position',[1000,500,240,120],'visible','off');
aa = uicontrol(handle, 'Style', 'PopupMenu', ...
    'Position', [20 60 200 40], ...
    'String', {'Enable automatic artifact detection','Disable automatic artifact detection'});
sb = uicontrol('Position',[20 20 200 40],'String','OK !','BackgroundColor',[ 0.702, 0.702, 0.702],'FontSize',10,...
    'FontWeight','bold','Callback','uiresume(gcbf)');
set(gcf,'visible','on');
uiwait(gcf);
artifactDetectionFlag = get(aa,'Value');
close(handle);
%% start waitbar
h=waitbar(0,'Please wait... calculating CSP features and classifier.','windowstyle', 'modal');
waitbar(0/100);
%%
fs = P_C.SamplingFrequency;
triggerTime = [2 6];
[~,~,NumChannels] = size(P_C.Data);
CueTimePoint = 3.5;
CueWindowNumber = CueTimePoint * 2;

if size(P_C.Data,1) == 1                 % data is not triggered yet
    %% try to read the Class Information out of the triggerChannel
    Markers = P_C.Marker;
    BeginMarkerSamples = Markers(Markers(:,3)==1,1);
    
    data_temp = P_C.Data;
    triggerData = data_temp(1,:,NumChannels);
    
    ClassInfo = triggerData(BeginMarkerSamples+1);
    if (length(ClassInfo) == 4 && ~isempty(find(ClassInfo ~= [-0.1 -0.2 -0.3 -0.4], 1 )))
        disp(['the runnumbers of the merged runs seem not to be "1,2,3,4"',' please check the data']);
    end
    clear data_temp;
    %%
    %Trigger
    New_tm{1}={NumChannels 1 'v' 0.9 0};
    SamplesBefore=triggerTime(1)*fs;
    SamplesAfter=triggerTime(2)*fs;
    Uncomplete=0;
    ChannelExclude=[];
    P_C_trig=gBStrigger(P_C,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);
    
    %% Load Class Information
    
    load classrun1_3class.mat;
    load classrun2_3class.mat;
    load classrun3_3class.mat;
    load classrun4_3class.mat;
    
    
    nTrialsPerRun = size(z1,2);
    nRuns = length(ClassInfo);
    class_info = zeros(size(z1,1),nRuns*nTrialsPerRun);
    for Nr = 1:length(ClassInfo)
        if ClassInfo(Nr) == -0.1
            zact = z1;
        elseif ClassInfo(Nr) == -0.2
            zact = z2;
        elseif ClassInfo(Nr) == -0.3
            zact = z3;
        elseif ClassInfo(Nr) == -0.4
            zact = z4;
        end
        class_info(:,(Nr-1)*nTrialsPerRun+1:Nr*nTrialsPerRun) = zact;
    end
    

    
    name_classes={
        'RIGHT'
        'LEFT'
        'FOOT'
        };
    
    use_rows=[1 2 3];
    P_C_trig=gBSloadclass(P_C_trig,class_info,name_classes,use_rows);
    waitbar(5/100);
    %% Cut triggerchannnel and timechannel
    ChannelExclude=[1  NumChannels];
    TrialExclude=[];
    P_C_trig=gBScuttrialschannels(P_C_trig,TrialExclude,ChannelExclude);
    
    %% Load the montage
    if NumChannels == 18
        MonFileName = 'CSP_Montage_16ch.mat';
    elseif NumChannels == 29
        MonFileName = 'CSP_Montage_27ch.mat';
    elseif NumChannels == 32
        MonFileName = 'CSP_Montage_30ch_Nautilus.mat';
    elseif NumChannels == 65
        MonFileName = 'CSP_Montage_63ch.mat';
    else
        MonFileName = '';
        disp('please load the montage file manually.');
    end
    
    
    if (~isempty(MonFileName) && (~isempty(which(MonFileName))))
        Mon = montage;
        Mon = load(Mon,MonFileName);
        
        P_C_trig.MontageName = Mon.MontageName;
        
        P_C_trig.XPosition = Mon.XPosition;
        P_C_trig.YPosition = Mon.YPosition;
        P_C_trig.ZPosition = Mon.ZPosition;
    end
    
    %% Filter
    bpFilter.Realization='butter';
    bpFilter.Type='BP';
    bpFilter.Order=5;
    bpFilter.f_high=30;
    bpFilter.f_low=8;
    FiltfiltFlag=0;
    TrialExclude=[];
    ChannelExclude=[];
    P_C_trig=gBSfilter(P_C_trig,bpFilter,FiltfiltFlag,ChannelExclude,TrialExclude);    
    %% save
    [~,~,NumChannels] = size(P_C_trig.Data);
else                    % data is already triggered
    P_C_trig = P_C;
end
waitbar(10/100);
%% part 2

global W_CSP_rightAll W_CSP_leftAll W_CSP_footAll;

windows = evalwindow;
%% Iterate over different windows
if size(windows,1)>1 %if more than one windows is selected => find that one with the best accuracy
    BestWindowErr = inf;
    for NW = 1:size(windows,1)
        
        P_C_S_copy = P_C_trig;
        
        clear T
        T=windows(NW,:).*fs;
        T(1) = T(1) + 1;            % start at sample '1' instead of '0'
        
        % variables for merging data
        Concatenate     = 'Trials';
        AdoptChAttr     = 1;
        AdoptTrialAttr  = 1;
        AdoptMarkers    = 1;
        name_classes    = {'ALL'};
        use_rows        = 1;
        
        %variables for apply CSP
        Replace         = 'replace all channels';
        FilterNumber    = [1   2  NumChannels-1  NumChannels];
        Transformation  = 'Create temporal pattern';
        
        
        
        %% Artifact Detection
        if  artifactDetectionFlag == 1
            [P_C_S_copy] = artifactDetection(P_C_S_copy, T, detectionThreshold, VarianceWindowLength);
        end
        
        %% cut out channels marked as bad and trials marked as artifact
        ChannelAttribute = P_C_S_copy.ChannelAttribute;
        ChannelAttributeName = P_C_S_copy.ChannelAttributeName;
        [~, indexBadChannel] = ismember('BAD',ChannelAttributeName);
        ChannelExclude = find(ChannelAttribute(indexBadChannel,:));
        AttributeNames = P_C_S_copy.AttributeName;
        Attribute       = P_C_S_copy.Attribute;
        [~, indexArtifact] = ismember('ARTIFACT',AttributeNames);
        TrialsExclude = find(Attribute(indexArtifact,:));
        
        P_C_S_copy=gBScuttrialschannels(P_C_S_copy,TrialsExclude,ChannelExclude);        
        %% cut out trials to get the same number of left and right trials
        
        AttributesNames = P_C_S_copy.AttributeName;
        Attribute       = P_C_S_copy.Attribute;
        for Nt = 1:size(AttributesNames,1)
            if ~isempty(strfind(AttributesNames{Nt},'RIGHT'));
                Nright = Nt;
            end
            if ~isempty(strfind(AttributesNames{Nt},'LEFT'));
                Nleft = Nt;
            end
            if ~isempty(strfind(AttributesNames{Nt},'FOOT'));
                Nfoot = Nt;
            end
        end
        
        Nall = [Nright,Nleft,Nfoot];
        
        numbersAll = [length(find(Attribute(Nright,:))),...
            length(find(Attribute(Nleft,:))),length(find(Attribute(Nfoot,:)))];
        minNumber = min(numbersAll);
        
        TrialExclude = [];
        for NC = 1:3
            if numbersAll(NC) > minNumber
                excess = numbersAll(NC) - minNumber;
                TrialExclude = [TrialExclude,find(Attribute(Nall(NC),:),excess,'first')];       % cannot be preallocated
            end
        end
        TrialExclude = unique(TrialExclude);
        
        if ~isempty(TrialExclude)
            P_C_S_copy=gBScuttrialschannels(P_C_S_copy,TrialExclude,ChannelExclude);
        end
        
        [NumTrials,NumSamples,NumChannels] = size(P_C_S_copy.Data);
        
        %% read out class_info
        Attribute = P_C_S_copy.Attribute;
        class_info = Attribute([Nright,Nleft,Nfoot],:);
        
        %% only right trials
        rightTrials = find((class_info(1,:)==1));
        TrialExclude = setdiff(1:NumTrials,rightTrials);
        P_C_S=gBScuttrialschannels(P_C_S_copy,TrialExclude,[]);
        save('temp_rightTrials','P_C_S'); 
        %% only left trials
        leftTrials = find((class_info(2,:)==1));
        TrialExclude = setdiff(1:NumTrials,leftTrials);
        P_C_S=gBScuttrialschannels(P_C_S_copy,TrialExclude,[]);
        save('temp_leftTrials','P_C_S'); 
        %% only foot trials
        footTrials = find((class_info(3,:)==1));
        TrialExclude = setdiff(1:NumTrials,footTrials);
        P_C_S=gBScuttrialschannels(P_C_S_copy,TrialExclude,[]);
        save('temp_footTrials','P_C_S'); 
        %% save data with all classes
        P_C_S = P_C_S_copy;
        save('temp_allTrials','P_C_S');
        clear P_C_S;
        
        %% CSP for left versus all
        
        % variables for CSP calculation
        TrialExclude    = [];
        ChannelExclude  = [];
        
        % merge data and apply classinfo for "ALL"
        P_C_n = data;
        P_C_n = load(P_C_n,'temp_allTrials.mat');
        class_infoRightFoot = zeros(1,length(class_info));
        class_infoRightFoot([rightTrials,footTrials]) = 1;                        % classinfo "ALL"
        P_C_n = gBSloadclass(P_C_n,class_infoRightFoot,name_classes,use_rows);
        FileName = {
            'temp_leftTrials.mat'
            };
        P_C_n = gBSmerge(P_C_n,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);
        
        % find class-numbers
        AttributesNames = P_C_n.AttributeName;
        for Nt = 1:size(AttributesNames,1)
            if ~isempty(strfind(AttributesNames{Nt},'RIGHT'));
                Nright = Nt;
            end
            if ~isempty(strfind(AttributesNames{Nt},'LEFT'));
                Nleft = Nt;
            end
            if ~isempty(strfind(AttributesNames{Nt},'FOOT'));
                Nfoot = Nt;
            end
            if ~isempty(strfind(AttributesNames{Nt},'ALL'));
                Nall = Nt;
            end
        end
        
        % calculate CSP
        Class1_nr=Nleft;
        Class2_nr=Nall;
        FileName='CSPleftAll.mat';
        C_O=gBScsp(P_C_n,T,Class1_nr,Class2_nr,TrialExclude,ChannelExclude,FileName,0);
        %     r2d = CreateResult2D(C_O);
        %     gResult2d(r2d);
        %     if NumChannels > 27
        %         plotOnly4CSPs(r2d);
        %     end
        %     set(gcf,'Name','CSP left versus all');
        clear C_O;
        
        % apply CSP filter on data
        SPF = spf;
        Filter = load(SPF,'CSPleftAll.mat');
        P_C_S = gBSspatialfilter(P_C_S_copy,Filter,FilterNumber,Replace,Transformation);
        save('temp_applied_leftAll','P_C_S');
        
        % Extract Weight-Matrix for CSPs
        spf_n = get(Filter,'spf');
        spf_struct = struct(spf_n);
        W_CSP_leftAll = spf_struct.D.W;
        
        clear P_C_n P_C_S
        %% CSP for right versus all
        
        % merge data and apply classinfo for "ALL"
        P_C_n = data;
        P_C_n = load(P_C_n,'temp_allTrials.mat');
        class_infoLeftFoot = zeros(1,length(class_info));
        class_infoLeftFoot([leftTrials,footTrials]) = 1;                          % classinfo "ALL"
        P_C_n = gBSloadclass(P_C_n,class_infoLeftFoot,name_classes,use_rows);
        FileName = {
            'temp_rightTrials.mat'
            };
        P_C_n = gBSmerge(P_C_n,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);
        
        % calculate CSP
        Class1_nr=Nright;
        Class2_nr=Nall;
        FileName='CSPrightAll.mat';
        C_O=gBScsp(P_C_n,T,Class1_nr,Class2_nr,TrialExclude,ChannelExclude,FileName,0);
        %     r2d = CreateResult2D(C_O);
        %     gResult2d(r2d);
        %     if NumChannels > 27
        %         plotOnly4CSPs(r2d);
        %     end
        %     set(gcf,'Name','CSP right versus all');
        clear C_O;
        
        % apply CSP filter on data
        SPF = spf;
        Filter = load(SPF,'CSPrightAll.mat');
        P_C_S= gBSspatialfilter(P_C_S_copy,Filter,FilterNumber,Replace,Transformation);
        save('temp_applied_rightAll','P_C_S');
        
        % Extract Weight-Matrix for CSPs
        spf_n = get(Filter,'spf');
        spf_struct = struct(spf_n);
        W_CSP_rightAll = spf_struct.D.W;
        
        clear P_C_n P_C_S
        %% CSP for foot versus all
        
        % merge data and apply classinfo for "ALL"
        P_C_n = data;
        P_C_n = load(P_C_n,'temp_allTrials.mat');
        class_infoLeftRight = zeros(1,length(class_info));
        class_infoLeftRight([leftTrials,rightTrials]) = 1;                        % classinfo "ALL"
        P_C_n = gBSloadclass(P_C_n,class_infoLeftRight,name_classes,use_rows);
        FileName = {
            'temp_footTrials.mat'
            };
        P_C_n = gBSmerge(P_C_n,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);
        
        % calculate CSP
        Class1_nr=Nfoot;
        Class2_nr=Nall;
        FileName='CSPfootAll.mat';
        C_O=gBScsp(P_C_n,T,Class1_nr,Class2_nr,TrialExclude,ChannelExclude,FileName,0);
        %     r2d = CreateResult2D(C_O);
        %     gResult2d(r2d);
        %     if NumChannels > 27
        %         plotOnly4CSPs(r2d);
        %     end
        %     set(gcf,'Name','CSP foot versus all');
        clear C_O;
        
        % apply CSP filter on data
        SPF = spf;
        Filter = load(SPF,'CSPfootAll.mat');
        P_C_S= gBSspatialfilter(P_C_S_copy,Filter,FilterNumber,Replace,Transformation);
        save('temp_applied_footAll','P_C_S');
        
        % Extract Weight-Matrix for CSPs
        spf_n = get(Filter,'spf');
        spf_struct = struct(spf_n);
        W_CSP_footAll = spf_struct.D.W;
        
        clear P_C_n P_C_S;
        %% Concatenate the three spatially filtered files
        P_C_sf = data;
        P_C_sf = load(P_C_sf,'temp_applied_rightAll.mat');
        FileName = {
            'temp_applied_leftAll.mat'
            'temp_applied_footAll.mat'
            };
        Concatenate='Channels';
        AdoptChAttr=1;
        AdoptTrialAttr=1;
        AdoptMarkers=1;
        P_C_sf = gBSmerge(P_C_sf,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);
        
        %% Variance
        ChannelExclude = [];
        IntervalLength = fs;
        GrowingWindow = 1;
        Overlap = fs-1;
        Replace = 'replace all channels';
        FileName = '';
        ProgressBarFlag = 0;
        P_C_sf= gBSvariance(P_C_sf, ChannelExclude, IntervalLength, GrowingWindow,...
            Overlap, Replace, FileName, ProgressBarFlag);
        
        %% Normalization
        ApplyOn = 'multiple channels';
        ChannelExclude_mult = [];
        TrialExclude_mult = [];
        Operation_mult = 'NORM';
        SecondOperand_mult(1) = 5;
        Unit_mult = 'µV';
        FirstOperand_two = 1;
        Operation_two = 'SUB';
        SecondOperand_two = 2;
        ProgressBarFlag = 0;
        P_C_sf = gBSarithmetic(P_C_sf, ApplyOn, ChannelExclude_mult,...
            TrialExclude_mult, Operation_mult, SecondOperand_mult,...
            Unit_mult, FirstOperand_two, Operation_two,...
            SecondOperand_two, ProgressBarFlag);
        
        %% Transform log
        ApplyOn = 'multiple channels';
        ChannelExclude_mult = [];
        TrialExclude_mult = [];
        Operation_mult = 'LOG10';
        SecondOperand_mult(1) = 5;
        Unit_mult = 'µV';
        FirstOperand_two = 1;
        Operation_two = 'SUB';
        SecondOperand_two = 2;
        ProgressBarFlag = 0;
        P_C_sf = gBSarithmetic(P_C_sf, ApplyOn, ChannelExclude_mult,...
            TrialExclude_mult, Operation_mult, SecondOperand_mult,...
            Unit_mult, FirstOperand_two, Operation_two,...
            SecondOperand_two, ProgressBarFlag);
        
        %% Feature Matrix
        Interval=[fs/2 fs/2 NumSamples];
        AttributeName={
            'RIGHT'
            'LEFT'
            'FOOT'
            };
        Permutate=0;
        MergeTimepoints = 0;
        ChannelExclude=[];
        FileName='';
        ProgressBarFlag=0;
        F_O=gBSfeaturematrix(P_C_sf,Interval,AttributeName,Permutate,MergeTimepoints,ChannelExclude,FileName,ProgressBarFlag);
        
        %% Linear Classifier
        Method='LDA';
        P.metric='';
        TrainTestData='CV';
        PlotFeatures=[1  4];
        FileName='';
        ProgressBarFlag=0;
        C_O=gBSlinearclassifier(F_O,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);        

        e = C_O.out_err;
        nErrors = length(e);
        err = zeros(nErrors,1);        
        for t = 1:nErrors
            err(t) =e{2,t}(1,1);
        end;
        
        min_err = sum(err(CueWindowNumber:end))/length(err(CueWindowNumber:end));  % minimum error
        if min_err < BestWindowErr
            BestWindowErr = min_err;
            takeWindow = T;
        end
        clear P_C_sf;
        waitbar((10+(50/size(evalwindow,1)*NW))/100);
    end
    disp(['selected window: ',num2str((takeWindow(1)-1)/fs),'s - ',num2str(takeWindow(2)/fs),'s ; ('...
        ,num2str((takeWindow(1)+1)),' samples - ',num2str(takeWindow(2)),' samples)']);
else
    takeWindow = windows.*fs;
end
%% after selecting the window for CSP do the whole calculation again...


%% Artifact Detection
if artifactDetectionFlag == 1
    [P_C_S_f] = artifactDetection(P_C_trig, takeWindow, detectionThreshold, VarianceWindowLength);
else
    P_C_S_f = P_C_trig;
end
% save data with selected artifacts and bad channels for later analysis
P_C_S = P_C_S_f;
save('CreateClassifier_DetectedArtifactInWindow.mat','P_C_S');
clear P_C_S;

%% cut out channels marked as bad and trials marked as artifact

ChannelAttribute = P_C_S_f.ChannelAttribute;
ChannelAttributeName = P_C_S_f.ChannelAttributeName;
[~, indexBadChannel] = ismember('BAD',ChannelAttributeName);
ChannelExclude = find(ChannelAttribute(indexBadChannel,:));

if ~isempty(ChannelExclude)
    disp(['The channel number(s): ',num2str(ChannelExclude),' have been removed for classifier setup.']);
else
    ChannelExclude = [];
end

AttributeNames = P_C_S_f.AttributeName;
Attribute       = P_C_S_f.Attribute;
[~, indexArtifact] = ismember('ARTIFACT',AttributeNames);
TrialsExclude = find(Attribute(indexArtifact,:));

if ~isempty(TrialsExclude)
    disp(['The trial number(s): ',num2str(TrialsExclude),' have been removed due to artifacts.']);
else
    TrialsExclude = [];
end

P_C_S_f=gBScuttrialschannels(P_C_S_f,TrialsExclude,ChannelExclude);

%% cut out trials to get the same number of left and right trials

AttributesNames = P_C_S_f.AttributeName;
Attribute       = P_C_S_f.Attribute;
for Nt = 1:size(AttributesNames,1)
    if ~isempty(strfind(AttributesNames{Nt},'RIGHT'));
        Nright = Nt;
    end
    if ~isempty(strfind(AttributesNames{Nt},'LEFT'));
        Nleft = Nt;
    end
    if ~isempty(strfind(AttributesNames{Nt},'FOOT'));
        Nfoot = Nt;
    end
end

Nall = [Nright,Nleft,Nfoot];

numbersAll = [length(find(Attribute(Nright,:))),...
    length(find(Attribute(Nleft,:))),length(find(Attribute(Nfoot,:)))];
minNumber = min(numbersAll);

TrialExclude = [];
for NC = 1:3
    if numbersAll(NC) > minNumber
        excess = numbersAll(NC) - minNumber;
        TrialExclude = [TrialExclude,find(Attribute(Nall(NC),:),excess,'first')];
    end
end
TrialExclude = unique(TrialExclude);

if ~isempty(TrialExclude)
    disp(['Removed trial(s) ',num2str(TrialExclude),' to get the same number of left and right trials.']);
    P_C_S_f=gBScuttrialschannels(P_C_S_f,TrialExclude,ChannelExclude);
end

[NumTrials,NumSamples,NumChannels] = size(P_C_S_f.Data);
waitbar(80/100);
%% read out class_info
Attribute = P_C_S_f.Attribute;
class_info = Attribute([Nright,Nleft,Nfoot],:);

%% only right trials
rightTrials = find((class_info(1,:)==1));
TrialExclude = setdiff(1:NumTrials,rightTrials);
P_C_S=gBScuttrialschannels(P_C_S_f,TrialExclude,[]);
save('temp_rightTrials','P_C_S');
%% only left trials
leftTrials = find((class_info(2,:)==1));
TrialExclude = setdiff(1:NumTrials,leftTrials);
P_C_S=gBScuttrialschannels(P_C_S_f,TrialExclude,[]);
save('temp_leftTrials','P_C_S');
%% only foot trials
footTrials = find((class_info(3,:)==1));
TrialExclude = setdiff(1:NumTrials,footTrials);
P_C_S=gBScuttrialschannels(P_C_S_f,TrialExclude,[]);
save('temp_footTrials','P_C_S');
%% save data with all classes
P_C_S = P_C_S_f;
save('temp_allTrials','P_C_S');
clear P_C_S;

%% variables for merging data
Concatenate     = 'Trials';
AdoptChAttr     = 1;
AdoptTrialAttr  = 1;
AdoptMarkers    = 1;
name_classes    = {'ALL'};
use_rows        = 1;

%variables for CSP calculation
TrialExclude    = [];
ChannelExclude  = [];
clear T;
T               = takeWindow;

%variables for apply CSP
Replace         = 'replace all channels';
FilterNumber    = [1   2  NumChannels-1  NumChannels];
Transformation  = 'Create temporal pattern';

%% CSP for left versus all
%merge data and apply classinfo for "ALL"
P_C_n = data;
P_C_n = load(P_C_n,'temp_allTrials.mat');
class_infoRightFoot = zeros(1,length(class_info));
class_infoRightFoot([rightTrials,footTrials]) = 1;                        % classinfo "ALL"
P_C_n = gBSloadclass(P_C_n,class_infoRightFoot,name_classes,use_rows);
FileName = {
    'temp_leftTrials.mat'
    };
P_C_n = gBSmerge(P_C_n,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);

%find class-numbers
AttributesNames = P_C_n.AttributeName;
Attribute       = P_C_n.Attribute;
for Nt = 1:size(AttributesNames,1)
    if ~isempty(strfind(AttributesNames{Nt},'RIGHT'));
        Nright = Nt;
    end
    if ~isempty(strfind(AttributesNames{Nt},'LEFT'));
        Nleft = Nt;
    end
    if ~isempty(strfind(AttributesNames{Nt},'FOOT'));
        Nfoot = Nt;
    end
    if ~isempty(strfind(AttributesNames{Nt},'ALL'));
        Nall = Nt;
    end
end

%calculate CSP
Class1_nr=Nleft;
Class2_nr=Nall;
FileName='CSPleftAll.mat';
C_O=gBScsp(P_C_n,T,Class1_nr,Class2_nr,TrialExclude,ChannelExclude,FileName,0);
r2d = CreateResult2D(C_O);
gResult2d(r2d);
if NumChannels > 27
    plotOnly4CSPs(r2d);
end
set(gcf,'Name','CSP left versus all');
clear C_O;

%apply CSP filter on data
SPF = spf;
Filter = load(SPF,'CSPleftAll.mat');
P_C_S = gBSspatialfilter(P_C_S_f,Filter,FilterNumber,Replace,Transformation);
save('temp_applied_leftAll','P_C_S');

%Extract Weight-Matrix for CSPs
spf_n = get(Filter,'spf');
spf_struct = struct(spf_n);
W_CSP_leftAll = spf_struct.D.W;
waitbar(90/100);
%% CSP for right versus all
%merge data and apply classinfo for "ALL"
P_C_n = data;
P_C_n = load(P_C_n,'temp_allTrials.mat');
class_infoLeftFoot = zeros(1,length(class_info));
class_infoLeftFoot([leftTrials,footTrials]) = 1;                          % classinfo "ALL"
P_C_n = gBSloadclass(P_C_n,class_infoLeftFoot,name_classes,use_rows);
FileName = {
    'temp_rightTrials.mat'
    };
P_C_n = gBSmerge(P_C_n,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);

%calculate CSP
Class1_nr=Nright;
Class2_nr=Nall;
FileName='CSPrightAll.mat';
C_O=gBScsp(P_C_n,T,Class1_nr,Class2_nr,TrialExclude,ChannelExclude,FileName,0);
r2d = CreateResult2D(C_O);
gResult2d(r2d);
if NumChannels > 27
    plotOnly4CSPs(r2d);
end
set(gcf,'Name','CSP right versus all');

%apply CSP filter on data
SPF = spf;
Filter = load(SPF,'CSPrightAll.mat');
P_C_S= gBSspatialfilter(P_C_S_f,Filter,FilterNumber,Replace,Transformation);
save('temp_applied_rightAll','P_C_S');

%Extract Weight-Matrix for CSPs
spf_n = get(Filter,'spf');
spf_struct = struct(spf_n);
W_CSP_rightAll = spf_struct.D.W;
% CSP for foot versus all

%merge data and apply classinfo for "ALL"
P_C_n = data;
P_C_n = load(P_C_n,'temp_allTrials.mat');
class_infoLeftRight = zeros(1,length(class_info));
class_infoLeftRight([leftTrials,rightTrials]) = 1;                        % classinfo "ALL"
P_C_n = gBSloadclass(P_C_n,class_infoLeftRight,name_classes,use_rows);
FileName = {
    'temp_footTrials.mat'
    };
P_C_n = gBSmerge(P_C_n,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);

%calculate CSP
Class1_nr=Nfoot;
Class2_nr=Nall;
FileName='CSPfootAll.mat';
C_O=gBScsp(P_C_n,T,Class1_nr,Class2_nr,TrialExclude,ChannelExclude,FileName,0);
r2d = CreateResult2D(C_O);
gResult2d(r2d);
if NumChannels > 27
    plotOnly4CSPs(r2d);
end
set(gcf,'Name','CSP foot versus all');

%apply CSP filter on data
SPF = spf;
Filter = load(SPF,'CSPfootAll.mat');
P_C_S= gBSspatialfilter(P_C_S_f,Filter,FilterNumber,Replace,Transformation);
save('temp_applied_footAll','P_C_S');
clear P_C_S_f P_C_n P_C_S;

%Extract Weight-Matrix for CSPs
spf_n = get(Filter,'spf');
spf_struct = struct(spf_n);
W_CSP_footAll = spf_struct.D.W;
%% Concatenate the three spatially filtered files
P_C_sf = data;
P_C_sf = load(P_C_sf,'temp_applied_rightAll.mat');
FileName = {
    'temp_applied_leftAll.mat'
    'temp_applied_footAll.mat'
    };
Concatenate='Channels';
AdoptChAttr=1;
AdoptTrialAttr=1;
AdoptMarkers=1;
P_C_sf = gBSmerge(P_C_sf,FileName,Concatenate,AdoptChAttr,AdoptTrialAttr,AdoptMarkers);
%% Variance
ChannelExclude = [];
IntervalLength = fs;
GrowingWindow = 1;
Overlap = fs-1;
Replace = 'replace all channels';
FileName = '';
ProgressBarFlag = 0;
P_C_sf= gBSvariance(P_C_sf, ChannelExclude, IntervalLength, GrowingWindow,...
    Overlap, Replace, FileName, ProgressBarFlag);
%% Normalization
ApplyOn = 'multiple channels';
ChannelExclude_mult = [];
TrialExclude_mult = [];
Operation_mult = 'NORM';
SecondOperand_mult(1) = 5;
Unit_mult = 'µV';
FirstOperand_two = 1;
Operation_two = 'SUB';
SecondOperand_two = 2;
ProgressBarFlag = 0;
P_C_sf = gBSarithmetic(P_C_sf, ApplyOn, ChannelExclude_mult,...
    TrialExclude_mult, Operation_mult, SecondOperand_mult,...
    Unit_mult, FirstOperand_two, Operation_two,...
    SecondOperand_two, ProgressBarFlag);
%% Transform log
ApplyOn = 'multiple channels';
ChannelExclude_mult = [];
TrialExclude_mult = [];
Operation_mult = 'LOG10';
SecondOperand_mult(1) = 5;
Unit_mult = 'µV';
FirstOperand_two = 1;
Operation_two = 'SUB';
SecondOperand_two = 2;
ProgressBarFlag = 0;
P_C_sf = gBSarithmetic(P_C_sf, ApplyOn, ChannelExclude_mult,...
    TrialExclude_mult, Operation_mult, SecondOperand_mult,...
    Unit_mult, FirstOperand_two, Operation_two,...
    SecondOperand_two, ProgressBarFlag);
%% Feature Matrix
Interval=[fs/2 fs/2 8*fs];
AttributeName={
    'RIGHT'
    'LEFT'
    'FOOT'
    };
ChannelExclude=[];
Permutate=0;
MergeTimepoints = 0;
FileName='fm_4var.mat';
ProgressBarFlag=0;
F_M=gBSfeaturematrix(P_C_sf,Interval,AttributeName,Permutate,MergeTimepoints,ChannelExclude,FileName,ProgressBarFlag);
%% Linear Classifier
PlotFeatures=[1  4];
Method='LDA';
P.metric='';
TrainTestData='100:100';
FileName='classifier_CSP_threeClasses.mat';
ProgressBarFlag=0;
C_O=gBSlinearclassifier(F_M,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);
r2d = CreateResult2D(C_O);
gResult2d(r2d);

%% Extract Weight-Matrix for CSPs
spf = get(Filter,'spf');
spf_struct = struct(spf);
W_CSP = spf_struct.D.W;

save W_CSP.mat W_CSP;
%% Display the result and store classificators for every step
out_wv  = C_O.out_clssfyr;
out_err = C_O.out_err;

%% new plot

e = C_O.out_err;
ErrorAllClasses = cell2mat(e(2,:));
ErrorTotal = ErrorAllClasses(1,CueWindowNumber:end);
ClassifierNumber = find(ErrorTotal == min(ErrorTotal));
nClassifiers = length(ClassifierNumber);

if nClassifiers > 1
       NeighborhoodValue = zeros(length(ClassifierNumber),1);
    for i=1:length(ClassifierNumber)
        Neighborhood = max(1,ClassifierNumber(i)-1):min(length(ErrorTotal),ClassifierNumber(i)+1);
        NeighborhoodValue(i) = sum(ErrorTotal(Neighborhood))/length(Neighborhood);
    end
    
    BestClassifierNumber = find(NeighborhoodValue == min(NeighborhoodValue));
    nBestClassifierNumber = length(BestClassifierNumber);
    
    if length(BestClassifierNumber) > 1
        GreaterNeighborhoodValue = zeros(nBestClassifierNumber,1);        
        for i=1:length(BestClassifierNumber)
            GreaterNeighborhood = max(1,ClassifierNumber(BestClassifierNumber(i))-2):min(length(ErrorTotal),ClassifierNumber(BestClassifierNumber(i))+2);
            GreaterNeighborhoodValue(i) = sum(ErrorTotal(GreaterNeighborhood))/length(GreaterNeighborhood);
        end
        
        BestBestClassifierNumber = find(GreaterNeighborhoodValue == min(GreaterNeighborhoodValue));
        
        % if still more than one best time point, take first one, otherwise
        % take the only one
        ClassifierNumber = ClassifierNumber(BestClassifierNumber(BestBestClassifierNumber(1)));
    else
        ClassifierNumber = ClassifierNumber(BestClassifierNumber);
    end
    
end

%Apply Classifier
ClassifierNumber=ClassifierNumber+CueWindowNumber-1;
Replace='replace all channels';
FileName=[pwd,filesep,'classification_result.mat'];
ProgressBarFlag=0;
ConfidenceInterval=[];
P_C_sf=gBSapplyclassifier(P_C_sf,C_O,ClassifierNumber,Replace,ConfidenceInterval,FileName,ProgressBarFlag);

%Classification Output Mapping
ClassIndex=[3 4 5];
ChannelExclude=[4 5 6 7];
TrialExclude=[];
FileName='';
SignificanceLevel=5;
ProgressBarFlag=1;
V_O = gBSclassificationoutputmapping(P_C_sf,ClassIndex,ChannelExclude,...
    TrialExclude,FileName,SignificanceLevel,ProgressBarFlag);

result2D = CreateResult2D(V_O);
gResult2d(result2D);
waitbar(100/100);
close(h);
clear P_C_S P_C_sf;
