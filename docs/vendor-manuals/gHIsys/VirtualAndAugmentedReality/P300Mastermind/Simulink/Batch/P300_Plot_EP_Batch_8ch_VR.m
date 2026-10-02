
global P_C

NumChannels = 8;
WindowLength = 800;
PreTriggerTime = 100;
DownSamplingFactor = 4;

% Artifact detection limits (Standard deviation criterion)
MinLimit = 2.5;
MaxLimit = 25;
Unit = 'µV';
TrialAttribute = 'ARTIFACT';

% Class Names

name_classes={
    
    'P300'
    'NP300'
};

old_P_C = P_C;

try
    % collect targets
    dt=old_P_C.Data;
    fl=diff(dt(:,:,NumChannels+3))>0;
    tg=find(diff(dt(:,:,end))>0);
    tg(end+1) = length(fl);
    nfl = zeros(1,length(tg)-1);
    for ntg=1:length(tg)-1
        nfl(ntg) = length(find(fl(tg(ntg):tg(ntg+1))));
    end
    pick = unique(nfl);
    if max(pick) < 5
        error('incomplete run, not trials');
    end
    cnt = zeros(1,length(pick));
    for pk=1:length(pick)
        cnt(pk) = length(find( nfl == pick(pk) ));
    end
    [NumSymbols,pk]=max(cnt);
    if NumSymbols < 3
        error('incomplete run not enough flashes');
    end
    NumTrialsPerSymbol = pick(pk);
    pick = find(nfl ~= NumTrialsPerSymbol);
    for cl = pick
        dt(:,tg(cl):tg(cl+1)-DownSamplingFactor,(NumChannels+2):end)=0;
    end
    tg(pick) = [];
    tg(tg==length(fl)) = [];
    P_C.Data = dt;
    Targets = dt(:,tg+1,size(dt,3));
    clear cnt cl pick fl dt
    
    samplingfrequency = P_C.SamplingFrequency;
    
    %Trigger
    New_tm{1}={NumChannels + 3, 1, 'v', 0.5, 1, 'FL', 'red'};
    SamplesBefore=ceil(P_C.SamplingFrequency * PreTriggerTime / 1000 );
    SamplesAfter=round(P_C.SamplingFrequency * WindowLength / 1000 ) - SamplesBefore;
    Uncomplete=0;
    ChannelExclude=[1,  NumChannels + 2, size(P_C.Data,3) ];
    P_C=gBStrigger(P_C,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);
    class_info=zeros(size(P_C.Data,1),2);
    ntrs = NumTrialsPerSymbol;
    curtarget = 1;
    dt = P_C.Data;
    for cl=1:size(class_info,1)
        tgr = find(dt(cl,:,NumChannels+1),1,'First');
        if ~isempty(find(dt(cl,tgr,NumChannels+2:dt(cl,tgr,NumChannels+1)+NumChannels+1) == Targets(curtarget), 1,'First'))
            class_info(cl,1) = 1;
        else
            class_info(cl,2) = 1;
        end
        ntrs = ntrs-1;
        if ntrs < 1
            curtarget = curtarget + 1;
            ntrs = NumTrialsPerSymbol;
        end
    end
    use_rows=1:size(class_info,2);
    P_C=gBSloadclass(P_C,class_info,name_classes,use_rows);
    clear dt cl ntrs curtarget 
    if P_C.SamplingFrequency == 250
        NewSmpFreq = 256;
        ProgressBarFlag = 0;
        P_C = gBSdownupsampling(P_C, NewSmpFreq, ProgressBarFlag);
    end
    
    % Signal quality check based on standard deviation
    IntervalStart = 1;
    IntervalEnd = size(P_C.Data,2);
    WindowLength = size(P_C.Data,2);
    ChannelAttribute = '';
    EpochName = '';
    TrialExclude = [];
    ChannelExclude = [NumChannels + 1:size(P_C.Data,3)];
    ProgressBarFlag = 0;
    P_C = gBSsqstandarddeviationcheck(P_C, IntervalStart, IntervalEnd, WindowLength,...
              MinLimit, MaxLimit, Unit, TrialAttribute, ChannelAttribute,...
              EpochName, TrialExclude, ChannelExclude, ProgressBarFlag);
    
    % deterend
    [~,id] = intersect(P_C.AttributeName,{TrialAttribute});
    attribute = P_C.Attribute;
    TrialExclude=find(attribute(id,:));
    ChannelExclude=[NumChannels+1:size(P_C.Data,3)];
    EstimationInterval = 'entire';
    StartInterval = [1];
    StopInterval = [205];
    TrendCorrection = 'individual';
    ProgressBarFlag = 0;
    [P_C, TrendCell] = gBSdetrend(P_C, TrialExclude, ChannelExclude,...
        EstimationInterval, StartInterval, StopInterval,...
        TrendCorrection, ProgressBarFlag);
    
    %Average
    Baseline=[1  SamplesBefore];
    Smoothing={'none'};
    DownSampling=0;
    FileName='';
    Averaging='different';
    Threshold=0.05;
    [~, ClassIndex]=intersect(P_C.AttributeName,name_classes);
    ClassIndex = sort(ClassIndex);
    A_O = gBSaverage(P_C,Baseline,Smoothing,DownSampling,TrialExclude,ChannelExclude,FileName,Averaging,0,ClassIndex,Threshold);
    r2d = CreateResult2D(A_O);
    gResult2d(r2d);
    a=1;
catch ME
    getReport(ME)
end
P_C = old_P_C;
