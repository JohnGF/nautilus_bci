% P300 Hyperscanning Speller Batch for gBSanalyze version 5.0
% g.tec medical engineering GmbH

% Use this batch for the data stored with gHyperscanning_MI_gHIamp.slx
% model
% run on postfixes: MI

global P_C;

%% General settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evalwindow = [                                          %evaluate the CSP in these windows
%     4.0 6.0;...
%     4.5 6.5;...    
%     5.0 7.0;...
%     5.5 7.5;...
%     6.0 8.0];
evalwindow = [5 7];                 
detectionThreshold = 10;                                 % threshold for artifact detection
VarianceWindowLength = 1.5;                             % length of window for calculation of variance (set the same value in the simulink model)

ClassOneName = 'Right';                                 % assign these names to the two classes
ClassTwoName = 'Left';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NumbWaitBarParts = size(evalwindow,1) + 2;              % to split up waitbar   
ClassNames={                                            % needed for gBSanalyze-commands
            ClassOneName
            ClassTwoName
};
%% enable or disable automatic artifact detection(create a classifier, apply a classifier, do both)
FigureHandle = figure('Name','Mode','Units','pixels','visible','off','MenuBar','none','IntegerHandle','off','NumberTitle','off');
set(FigureHandle,'Position',[1000,500,240,120],'visible','off');
aa = uicontrol(FigureHandle, 'Style', 'PopupMenu', ...
    'Position', [20 60 200 40], ...
    'String', {'Enable automatic artifact detection','Disable automatic artifact detection'});
sb = uicontrol('Position',[20 20 200 40],'String','OK !','BackgroundColor',[ 0.702, 0.702, 0.702],'FontSize',10,...
    'FontWeight','bold','Callback','uiresume(gcbf)');
set(FigureHandle,'visible','on');
uiwait(FigureHandle);
artifactDetectionFlag = get(aa,'Value');
close(FigureHandle);
%% start waitbar
h=waitbar(0,'Please wait... calculating CSP features and classifier.','windowstyle', 'modal');
uistack(h,'top');


try
waitbar(0/100);
WaitBarCounter = 0;

fs = P_C.SamplingFrequency;
triggerTime = [2 6];        
[NumTrials,~,NumChannels] = size(P_C.Data);
CueTimePoint = 3.5;                             
CueWindowNumber = CueTimePoint * 2; 
if NumTrials == 1                                           % data is not triggered yet        
    %% try to read the Class Information out of the triggerChannel
    Markers = P_C.Marker;
    BeginMarkerSamples = Markers(Markers(:,3)==1,1);
    
    data_temp = P_C.Data;
    triggerData = data_temp(1,:,NumChannels);
    
    ClassInfo = triggerData(BeginMarkerSamples+1);
    if (length(ClassInfo) == 4 && ~isempty(find(ClassInfo ~= [-0.1 -0.2 -0.3 -0.4], 1 )))
        disp(['the runnumbers of the merged runs seem not to be "1,2,3,4". ','please check the data']);
    end
    clear data_temp;
    %%
    %Trigger
    New_tm{1}={NumChannels 1 'v' 0.9 0};
    SamplesBefore=triggerTime(1)*fs;
    SamplesAfter=triggerTime(2)*fs;
    Uncomplete=0;
    ChannelExclude=[];
    P_C_triggered=gBStrigger(P_C,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);
    %% Load Class Information    
    load classrun1.mat;
    load classrun2.mat;
    load classrun3.mat;
    load classrun4.mat;
    
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
    
    
    use_rows=[1  2];
    P_C_triggered=gBSloadclass(P_C_triggered,class_info,ClassNames,use_rows);
    
    %% Cut triggerchannnel and timechannel
    ChannelExclude=[1  NumChannels];
    TrialExclude=[];
    P_C_triggered=gBScuttrialschannels(P_C_triggered,TrialExclude,ChannelExclude);    
          
    %% Filter    
    bpFilter.Realization='butter';
    bpFilter.Type='BP';
    bpFilter.Order=5;
    bpFilter.f_high=30;
    bpFilter.f_low=8;
    FiltfiltFlag=0;
    TrialExclude=[];
    ChannelExclude=[];
    P_C_triggered=gBSfilter(P_C_triggered,bpFilter,FiltfiltFlag,ChannelExclude,TrialExclude);    

else                                    % data is already triggered
    P_C_triggered = P_C;            
end
%% save data with selected artifacts and bad channels for later analysis
P_C_S = P_C_triggered;
clear P_C_triggered;
WaitBarCounter = WaitBarCounter + 1;
waitbar(WaitBarCounter/NumbWaitBarParts);
%% cut out channels marked as bad
ChannelAttribute = P_C_S.ChannelAttribute;
ChannelAttributeName = P_C_S.ChannelAttributeName;
[~, indexBadChannel] = ismember('BAD',ChannelAttributeName);
ChannelExclude = find(ChannelAttribute(indexBadChannel,:));
P_C_S = gBScuttrialschannels(P_C_S,[],ChannelExclude);

[~,NumSamplesTriggered,NumChannelsTriggered] = size(P_C_S.Data);
%% part 2
global W_CSP;
%% Iterate over different windows
if size(evalwindow,1)>1    % if more than one windows is selected => find that one with the best accuracy
    BestWindowErr = inf;
    for NW = 1:size(evalwindow,1)
        P_C_S_copy = P_C_S;
        
        clear T;
        Class1_nr=3;
        Class2_nr=4;
        T=evalwindow(NW,:).*fs;
        T(1) = T(1) + 1;            % start at sample '1' instead of '0'
        
        WBstep = (NW/(size(evalwindow,1)))*0.65 + 15/100;        
        %% Artifact Detection
        if artifactDetectionFlag == 1
            [P_C_S_copy] = artifactDetection(P_C_S_copy, T, detectionThreshold, VarianceWindowLength);
        end
        %%  cut trial marked as artifact
        AttributeNames = P_C_S_copy.AttributeName;
        Attribute       = P_C_S_copy.Attribute;
        [~, indexArtifact] = ismember('ARTIFACT',AttributeNames);
        TrialExclude = find(Attribute(indexArtifact,:));
        P_C_S_copy=gBScuttrialschannels(P_C_S_copy,TrialExclude,[]);       
        [~,~,NumChannelsTriggered] = size(P_C_S.Data);
        %% cut out trials to get the same number of left and right trials                
        ChannelExclude = [];
        AttributesNames = P_C_S_copy.AttributeName;
        Attribute       = P_C_S_copy.Attribute;
        for Nt = 1:size(AttributesNames,1)
            if ~isempty(strfind(AttributesNames{Nt},ClassOneName));
                Nright = Nt;
            end
            if ~isempty(strfind(AttributesNames{Nt},ClassTwoName));
                Nleft = Nt;
            end
        end
        
        numbersLeft = length(find(Attribute(Nleft,:)));
        numbersRight = length(find(Attribute(Nright,:)));
        
        deleteTrials = numbersRight - numbersLeft;
        if deleteTrials > 0 %delete right trials
            TrialExclude = find(Attribute(Nright,:),deleteTrials,'first');
        elseif deleteTrials < 0 %delete left trials
            TrialExclude = find(Attribute(Nleft,:),abs(deleteTrials),'first');          
        else
            TrialExclude = [];
        end              
        %% cut trials and channels
         P_C_S_copy = gBScuttrialschannels(P_C_S_copy,TrialExclude,ChannelExclude);        
        %% CSP
        CSPFileName='createdCSP.mat';
        csp=gBScsp(P_C_S_copy,T,Class1_nr,Class2_nr,[],[],CSPFileName,0);
        %% Spatial Filter
        SPF=spf;
        Filter=load(SPF,CSPFileName);
        FilterNumber=[1   2  NumChannelsTriggered-1  NumChannelsTriggered];
        Replace='replace all channels';
        Transformation='Create temporal pattern';
        P_C_S_copy=gBSspatialfilter(P_C_S_copy,Filter,FilterNumber,Replace,Transformation);
        %% Variance
        ChannelExclude = [];
        IntervalLength = fs;
        GrowingWindow = 1;
        Overlap = fs-1;
        Replace = 'replace all channels';
        FileName = '';
        ProgressBarFlag = 0;
        P_C_S_copy = gBSvariance(P_C_S_copy, ChannelExclude, IntervalLength, GrowingWindow,...
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
        P_C_S_copy = gBSarithmetic(P_C_S_copy, ApplyOn, ChannelExclude_mult,...
            TrialExclude_mult, Operation_mult, SecondOperand_mult,...
            Unit_mult, FirstOperand_two, Operation_two,...
            SecondOperand_two, ProgressBarFlag);
        %% Transform
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
        P_C_S_copy = gBSarithmetic(P_C_S_copy, ApplyOn, ChannelExclude_mult,...
            TrialExclude_mult, Operation_mult, SecondOperand_mult,...
            Unit_mult, FirstOperand_two, Operation_two,...
            SecondOperand_two, ProgressBarFlag);
        %% Feature Matrix
        Interval = [fs/2,fs/2,NumSamplesTriggered];       
        Permutate=0;
        MergeTimePoints=0;
        ChannelExclude=[];
        FileName='';
        ProgressBarFlag=0;
        F_M = gBSfeaturematrix(P_C_S_copy,Interval,ClassNames,Permutate,MergeTimePoints,ChannelExclude,FileName,ProgressBarFlag);
        %% Linear Classifier CV
        Method='LDA';
        P.metric='';
        TrainTestData='CV';
        PlotFeatures=[1  4];
        FileName='';
        ProgressBarFlag=0;
        C_O = gBSlinearclassifier(F_M,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);
        
        err = [];
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
        WaitBarCounter = WaitBarCounter + 1; 
        waitbar(WaitBarCounter/NumbWaitBarParts);
    end
    disp(['selected window: ',num2str((takeWindow(1)-1)/fs),'s - ',num2str(takeWindow(2)/fs),'s ; ('...
        ,num2str(takeWindow(1)-1),' samples - ',num2str(takeWindow(2)),' samples)']);
else
    takeWindow = evalwindow.*fs;
end
clear P_C_S_copy;
clear T
Class1_nr=3;
Class2_nr=4;
T=takeWindow;
%% Artifact Detection
if artifactDetectionFlag == 1
    [P_C_S] = artifactDetection(P_C_S, T, detectionThreshold, VarianceWindowLength);
end
P_CPreprocessed = P_C_S;
% save data with selected artifacts and bad channels for later analysis
save('CreateClassifier_DetectedArtifactInWindow.mat','P_C_S');
%% cut out channels marked as bad and trials marked as artifact
ChannelAttribute = P_C_S.ChannelAttribute;
ChannelAttributeName = P_C_S.ChannelAttributeName;
[~, indexBadChannel] = ismember('BAD',ChannelAttributeName);
ChannelExclude = find(ChannelAttribute(indexBadChannel,:));

if ~isempty(ChannelExclude)
    disp(['The channel number(s): ',num2str(ChannelExclude),' have been removed for classifier setup.']);
else
    ChannelExclude = [];
end

AttributeNames = P_C_S.AttributeName;
Attribute       = P_C_S.Attribute;
[~, indexArtifact] = ismember('ARTIFACT',AttributeNames);
TrialExclude = find(Attribute(indexArtifact,:));

if ~isempty(TrialExclude)
    disp(['The trial number(s): ',num2str(TrialExclude),' have been removed due to artifacts.']);
else
    TrialExclude = [];
end

P_C_S = gBScuttrialschannels(P_C_S,TrialExclude,ChannelExclude);
%% cut out trials to get the same number of left and right trials
AttributesNames = P_C_S.AttributeName;
Attribute       = P_C_S.Attribute;
for Nt = 1:size(AttributesNames,1)
    if ~isempty(strfind(AttributesNames{Nt},ClassOneName));
        Nright = Nt;
    end
    if ~isempty(strfind(AttributesNames{Nt},ClassTwoName));
        Nleft = Nt;
    end
end

numbersLeft = length(find(Attribute(Nleft,:)));
numbersRight = length(find(Attribute(Nright,:)));

deleteTrials = numbersRight - numbersLeft;
if deleteTrials > 0 %delete right trials
    TrialExclude = find(Attribute(Nright,:),deleteTrials,'first');
elseif deleteTrials < 0 %delete left trials
    TrialExclude = find(Attribute(Nleft,:),abs(deleteTrials),'first');
else
    TrialExclude = [];
end

if ~isempty(TrialExclude)
    disp(['Removed trial(s) ',num2str(TrialExclude),' to get the same number of left and right trials.']);
end
%% CSP
ChannelExclude = [];
CSPFileName='createdCSP.mat';
C_O=gBScsp(P_C_S,T,Class1_nr,Class2_nr,TrialExclude,ChannelExclude,CSPFileName,0);
r2dCSP = CreateResult2D(C_O);     
%% Spatial Filter
SPF=spf;
Filter=load(SPF,CSPFileName);
FilterNumber=[1   2  NumChannelsTriggered-1  NumChannelsTriggered];
Replace='replace all channels';
Transformation='Create temporal pattern';
P_C_S=gBSspatialfilter(P_C_S,Filter,FilterNumber,Replace,Transformation);
%% Variance
ChannelExclude = [];
IntervalLength = VarianceWindowLength*fs;
GrowingWindow = 1;
Overlap = IntervalLength -1;
Replace = 'replace all channels';
FileName = '';
ProgressBarFlag = 0;
P_C_S = gBSvariance(P_C_S, ChannelExclude, IntervalLength, GrowingWindow,...
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
P_C_S = gBSarithmetic(P_C_S, ApplyOn, ChannelExclude_mult,...
    TrialExclude_mult, Operation_mult, SecondOperand_mult,...
    Unit_mult, FirstOperand_two, Operation_two,...
    SecondOperand_two, ProgressBarFlag);
%% Transform
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
P_C_S = gBSarithmetic(P_C_S, ApplyOn, ChannelExclude_mult,...
    TrialExclude_mult, Operation_mult, SecondOperand_mult,...
    Unit_mult, FirstOperand_two, Operation_two,...
    SecondOperand_two, ProgressBarFlag);
%% Feature Matrix
Interval = [fs/2,fs/2,NumSamplesTriggered];
ChannelExclude=[];
Permutate=0;
MergeTimePoints=0;
FileName='';
ProgressBarFlag=0;
F_M=gBSfeaturematrix(P_C_S,Interval,ClassNames,Permutate,MergeTimePoints,ChannelExclude,FileName,ProgressBarFlag);
%% Linear Classifier
PlotFeatures=[1  4];
Method='LDA';
P.metric='';
TrainTestData='100:100';
FileName='Hyperscanning_MI_classifier.mat';
ProgressBarFlag=0;
C_O=gBSlinearclassifier(F_M,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);

r2dClassifier = CreateResult2D(C_O);
%% Extract Weight-Matrix for CSPs
spf = get(Filter,'spf');
spf_struct = struct(spf);
W_CSP = spf_struct.D.W;
save W_CSP.mat W_CSP;
%% Display the result and store classificators for every step
out_wv  = C_O.out_clssfyr;
out_err = C_O.out_err;

nErrorVals = size(out_err,2);
out_err_ = zeros(nErrorVals,1);
out_err_x = zeros(nErrorVals,2);
for i=1:nErrorVals
    out_err_temp = out_err{2,i};
    out_err_(i) = out_err_temp(1);
    out_err_x(i,1:2) = out_err{1,i};
end
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
   
   if nBestClassifierNumber > 1
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
P_C_S=gBSapplyclassifier(P_C_S,C_O,ClassifierNumber,Replace,ConfidenceInterval,FileName,ProgressBarFlag);

%Classification Output Mapping 
ClassIndex=[3 4]; 
ChannelExclude=[3 4 5]; 
TrialExclude=[];
FileName='';
SignificanceLevel=5; 
ProgressBarFlag=1; 
V_O = gBSclassificationoutputmapping(P_C_S,ClassIndex,ChannelExclude,... 
    TrialExclude,FileName,SignificanceLevel,ProgressBarFlag); 
r2dMapping = CreateResult2D(V_O);
gResult2d(r2dMapping);
%% Figure options
f = figure('Position',[3 30 1000 660],'Name','BCI Experiment','NumberTitle','off'); 
x = out_err_x(:,1);
y = out_err_;
a = axes('Position',[0.1 0.25 0.4 0.4]);

[value,index] = min(out_err_);
plot(x,y, x(index),y(index),'or','Parent',a);
set(a,'YLim',[0 50]);
ylabel('Error rate [%]');
xlabel('time [sec]');
name = P_C_S.SubjectLastName;
tmp  = ['title(','''','Classification Error: ','''',');'];
grid on
maximum  = max(get(a,'XLim'));
minimumX = min(get(a,'XLim'));
text(maximum+0.2,-2,'EXCELLENT','Rotation',90);
text(maximum+0.2,16,'GOOD','Rotation',90);
line([minimumX maximum],[10 10],'Color','red');
line([minimumX maximum],[30 30],'Color','black');
text(maximum+0.2,31,'MORE TRAINING','Rotation',90);
eval(tmp)

i=85;
d=3;

t=text(1,i,'Brain-Computer Interface Hyperscanning Experiment: Demo','FontSize',18,'FontWeight','bold');
i=i-d-d;
t=text(1,i,'An Electroencephalogram-based Brain-Computer Interface (EEG-based BCI) provides a new communication','FontSize',10);
i=i-d;
t=text(1,i,'channel between the human brain and the computer. Patients who suffer from severe motor impairments');
i=i-d;
t=text(1,i,'(e.g. late stage of Amyotrophic Lateral Sclerosis (ALS), severe cerebral palsy, head trauma and spinal');
i=i-d;
text(1,i,'injuries) use such a BCI system as an alternative form of communication controlled by mental activity.');

i=i-d;
i=-10;
text(1,i,'A modern BCI enables fast and easy implementation of different processing algorithms and classification');
i=i-d;
text(1,i,'methods for optimal classification accuracy. Therefore, this new BCI uses the g.tec rapid prototyping');
i=i-d;
text(1,i,'environment to enable a fast transfer of specific EEG-analysis algorithms to real-time implementation.');
i=i-d;
text(1,i,'The system allows you to achieve reliable results in an early stage of development and to perform a rapid');
i=i-d;
text(1,i,'iteration of the design.');
i=i-d-d;

t=text(1,i,'Realized with g.HIamp and g.BSanalyze.','FontSize',10,'FontWeight','bold');

b = axes; 
set(b,'Position', [0.1300+0.4    0.2500    0.7750/2.05    0.8150/2.05]);
[A,B] = imread('paradigm.gif');
image(A,'Parent',b);
colormap(B)
axis off

clear P_C_S;



% plotting CSP for each user
% detecting the number of users
NrOfCh = size(P_CPreprocessed.Data,3);
if isequal (NrOfCh/63,round(NrOfCh/63))
    users = NrOfCh/63;
    MonFileName = 'CSP_Montage_63ch.mat';
    ChPerUser = 63;
elseif isequal (NrOfCh/27,round(NrOfCh/27))
    users = NrOfCh/27;
    MonFileName = 'CSP_Montage_27ch.mat';
    ChPerUser = 27;
elseif isequal (NrOfCh/16,round(NrOfCh/16))
    users = NrOfCh/16;
    MonFileName = 'CSP_Montage_16ch.mat';
    ChPerUser = 16;
end

if exist('ChPerUser','var')
    UserCh = {};
    for userNumber = 1:users
        UserCh{userNumber} = (userNumber-1)*ChPerUser+1:userNumber*ChPerUser;
    end
end
%% Extracting the data for each user
if exist('users','var')
    for userNr = 1:users
        wu = waitbar(userNr/users,'plotting the spatial pattern for each user','WindowStyle','modal');        
        ChannelExclude = setdiff(P_CPreprocessed.Channels,UserCh{userNr});
        userTemp = gBScuttrialschannels(P_CPreprocessed,[],ChannelExclude);
        if (~isempty(MonFileName) && (~isempty(which(MonFileName))))
            Mon = montage;
            Mon = load(Mon,MonFileName);
            userTemp.MontageName = Mon.MontageName;
            userTemp.XPosition = Mon.XPosition;
            userTemp.YPosition = Mon.YPosition;
            userTemp.ZPosition = Mon.ZPosition;
        end        
        CSPFileName='createdCSP.mat';
        C_O=gBScsp(userTemp,T,Class1_nr,Class2_nr,TrialExclude,[],CSPFileName,0);
        r2dCSP = CreateResult2D(C_O);
        gResult2d(r2dCSP);
        close(wu);        
    end
    clear userTemp;
end
clear P_CPreprocessed;


waitbar(1,h);
catch
    troubles('problems creating the spatial patterns and Classifier','Please check if the correct file is loaded');    
end
close(h);
