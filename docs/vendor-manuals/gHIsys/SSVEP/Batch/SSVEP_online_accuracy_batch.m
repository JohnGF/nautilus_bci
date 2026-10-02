
%Accuracy for Online Classification
% run on postfixes: SSVEP
%##################################

global P_C

sampFrequency = P_C.SamplingFrequency;

%Classinfo File
%--------------
classinfo = 'classinfo_20tr.m';

%Trigger-settings
%------------------
TriggerLow = 3;
TriggerHigh = 7;
SamplesBefore = round(TriggerLow*sampFrequency); 
SamplesAfter = round(TriggerHigh*sampFrequency);


old_PC = P_C;

try
    
    classfile = abs(load(classinfo));
    class_ = zeros(1,size(classfile,1));
    for cntTrial=1:size(classfile,1)
        nextclass = find(classfile(cntTrial,:),1,'first');
        if isempty(nextclass)
            nextclass = 0;
        end
        class_(cntTrial) = nextclass;
    end
    
    %Trigger
    if size(old_PC.Data,3)==11
        New_tm{1}={10 1 'v' 0.9 0 'TRIG' 'red'};
        Uncomplete=0;
        ChannelExclude=[1   2   3   4   5   6   7   8   9  10];
    else
        New_tm{1}={9 1 'v' 0.9 0 'TRIG' 'red'};
        Uncomplete=0;
        ChannelExclude=[1   2   3   4   5   6   7   8  9];
    end
    P_C=gBStrigger(P_C,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);
    
    classResult = P_C.Data;
    if    ( min(min(classResult)) == 0 ) ...
       && ( length(unique(classResult)) == length(unique(class_)) )
        % old data set where class ids start with zero instead of 1
        classResult = classResult + 1;
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
    error_ = (1-result(1,:) ).*100;
    fp = result(2,:) .* 100;
    
    time = linspace(0,TriggerLow+TriggerHigh,size(error_,2));
    
    figure('Name','SSVEP Online Accuracy')
    plot(time,[ error_; fp]);
    ylim([0 100]);
    line([3 3],[0 100],'LineWidth',4,'Color','r');
    legend('Error','False Positiv')
    xlabel('time [s]');
    ylabel('Classification Error/False Positiv [%]')
    title('Online accuracy')
    
catch err
    disp(err);
    disp(err.stack(:))
    P_C = old_PC;
end