% BCI Batch for gBSanalyze version 4.12.01
% 2013 - g.tec medical engineering GmbH


clear;
global P_C
% global V_R


P_C_copy = P_C;
% Simulink model with 256 Hz
fs=P_C.SamplingFrequency;
[NumTrials,NumSamples,NumChannels] = size(P_C.Data);


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

y_dat = P_C_copy.Data;
if (size(y_dat,3) == 4)
    TrialExclude=[];
    ChannelExclude=[1];
    P_C_copy=gBScuttrialschannels(P_C_copy,TrialExclude,ChannelExclude);
end


% Calculating the post-trigger time

trig_ch = y_dat(1,:,size(y_dat,3));
triggerTimes = find(diff(trig_ch) > 0.5);
intertrialInterval = diff([triggerTimes(1),triggerTimes(2)]);
clear y_dat;

% Trigger the data, 2 seconds before trigger and 6 seconds after trigger
New_tm{1}={3 1 'l' 90 0};
SamplesBefore=2*fs;
SamplesAfter=intertrialInterval - 3*fs;
Uncomplete=0;
ChannelExclude=[];
P_C_copy=gBStrigger(P_C_copy,New_tm,SamplesBefore,SamplesAfter,Uncomplete,ChannelExclude);

%% Load Class Information

load classrun1.mat;
load classrun2.mat;
load classrun3.mat;
load classrun4.mat;

class_info = [];
for Nr = 1:length(ClassInfo)
    if ClassInfo(Nr) == -0.1
        class_info = [class_info,z1];
    elseif ClassInfo(Nr) == -0.2
        class_info = [class_info,z2];
    elseif ClassInfo(Nr) == -0.3
        class_info = [class_info,z3];
    elseif ClassInfo(Nr) == -0.4
        class_info = [class_info,z4];         
    else
        troubles('Class info could not be loaded','make sure that the data was recorded correctly');
        return
    end
end

name_classes={
    'RIGHT'
    'LEFT'
};

use_rows=[1  2];
P_C_copy=gBSloadclass(P_C_copy,class_info,name_classes,use_rows);
%%

% Bandpower in alpha range
ChannelExclude = [3];
Filter.Name = 'ALPHA_BCI';
Filter.Type = 'BP';
Filter.f_low = [8];
Filter.f_high = [12];
Filter.Realization = 'butter';
Filter.Order = [4];
IntervalLength = fs;
Overlap = fs-1;
Replace = 'add channels';
FileName = '';
ProgressBarFlag = 0;
P_C_copy = gBSbandpower(P_C_copy, ChannelExclude, Filter, IntervalLength,...
    Overlap, Replace, FileName, ProgressBarFlag);

% Bandpower in beta range
ChannelExclude = [3  4  5];
Filter.Name = 'BETA_BCI';
Filter.Type = 'BP';
Filter.f_low = [16];
Filter.f_high = [24];
Filter.Realization = 'butter';
Filter.Order = [4];
IntervalLength = fs;
Overlap = fs-1;
Replace = 'add channels';
FileName = '';
ProgressBarFlag = 0;
P_C_copy = gBSbandpower(P_C_copy, ChannelExclude, Filter, IntervalLength,...
    Overlap, Replace, FileName, ProgressBarFlag);

% Transform log
ApplyOn = 'multiple channels';
ChannelExclude_mult = [1  2  3];
TrialExclude_mult = [];
Operation_mult = 'LOG10';
SecondOperand_mult(1) = 5;
Unit_mult = 'µV';
FirstOperand_two = 1;
Operation_two = 'SUB';
SecondOperand_two = [2];
ProgressBarFlag = 0;
P_C_copy = gBSarithmetic(P_C_copy, ApplyOn, ChannelExclude_mult,...
    TrialExclude_mult, Operation_mult, SecondOperand_mult,...
    Unit_mult, FirstOperand_two, Operation_two,...
    SecondOperand_two, ProgressBarFlag); 

% % Visualize the new data
% [V_R]=plot(P_C_copy,V_R);

% Feature Matrix
Interval=[fs   fs/2  SamplesAfter + SamplesBefore];
AttributeName={
    'RIGHT'
    'LEFT'
};
ChannelExclude=[1  2  3];
Permutate=0;
MergeTimepoints=0;
FileName=[''];
ProgressBarFlag=[0];
F_M=gBSfeaturematrix(P_C_copy,Interval,AttributeName,Permutate,MergeTimepoints,ChannelExclude,FileName,ProgressBarFlag);

% Linear Classifier
PlotFeatures=[1  2];
Method=['LDA'];
P.metric=[''];
TrainTestData=['100:100'];
FileName=['CreatedClassifier'];
ProgressBarFlag=[0];
C_O=gBSlinearclassifier(F_M,Method,P,TrainTestData,PlotFeatures,FileName,ProgressBarFlag);

% Storage of weight vectors
e = C_O.out_err;
err = zeros(1,length(e));
for t=1:length(e)
    v = e(2,t);
    err(1,t) = v{1}(1);
end

% Index of minimum error
index = find(err == min(err));
index = index(end);

% Extract the data from C_O_S and save it
% File: weight vector, bias
fid=fopen('Bci_weights.txt','w');

w = C_O.out_clssfyr;
m = w(index,:);
weights = zeros(1,size(m{2},1));
for u=1:length(m{2})
    s = m{2}(u);
    weights(1,u) = s;
    fprintf(fid,'%6.14f,',s);
end

% Save the bias value
bias = m{1};
fprintf(fid,'%6.14f',bias(1));
fclose(fid);
% End of vector storage

% Display the result
out_wv=C_O.out_clssfyr
out_err=C_O.out_err

tmp4=TrainTestData;
fid=fopen('classification.dat','w');
fprintf(fid,'Classification Method: Linear Discriminant Analysis');
fprintf(fid,'\n');
tmp=['fprintf(fid,','''Training- and testdata option: ',tmp4,'\n''',');'];
eval(tmp);
attrname=P_C_copy.AttributeName;
out_err_ = zeros(1,size(out_err,2));
out_err_x = zeros(size(out_err,2),2);
for i=1:size(out_err,2)
    out_err_(i)=out_err{2,i}(1);
    out_err_x(i,1:2)=out_err{1,i};
    fprintf(fid,'%6.1f ',out_err{2,i}(1));
    fprintf(fid,'\n');
end
if (~isempty(out_wv))
    fprintf(fid,'\n');
    fprintf(fid,'Weight Vector: ');
    fprintf(fid,'\n');
    fprintf(fid,'Bias, WV Par 1, WV Par 2,...');
    fprintf(fid,'\n');
    out_wv_ = zeros(size(out_wv,1),5);
    for i=1:size(out_wv,1)
        out_wv_(i,:) = [out_wv{i,1}(1) out_wv{i,2}(1:4)]; 
        fprintf(fid,'%6.6f ',[out_wv{i,1}(1) out_wv{i,2}(1:4)]);
        fprintf(fid,'\n');
    end
end
        
status=fclose(fid);
[minimum,index]=min(out_err_);
WV=out_wv_(index,:);

% P_C_copy_S=P_C_copy;
[Numtrials,~,~] = size(P_C_copy.Data);

minimum=min(out_err_);
anzahl=0;
mini=[];

if (exist('minimumsave.mat','file') == 2)
    load minimumsave.mat
end
mini=[mini minimum];
bessere=length(find(mini<minimum));
anzahl=anzahl+1;
save minimumsave mini anzahl
%%
f=figure('Position',[3 30 1000 660],'Name','BCI Experiment','NumberTitle','off'); 
x=out_err_x(:,1);
y=out_err_;
a=axes('Position',[0.1 0.25 0.4 0.4]);

[value,index]=min(out_err_);
plot(x,y, x(index),y(index),'or','Parent',a);
set(a,'YLim',[0 50]);
ylabel('Error rate [%]');
xlabel('time [sec]');
name=P_C_copy.SubjectLastName;
tmp=['title(','''','Classification Error: ','''',');'];
grid on
maximum=max(get(a,'XLim'));
minimumX=min(get(a,'XLim'));
text(maximum+0.2,-2,'EXCELLENT','Rotation',90);
text(maximum+0.2,16,'GOOD','Rotation',90);
line([minimumX maximum],[10 10],'Color','red');
line([minimumX maximum],[30 30],'Color','black');
text(maximum+0.2,31,'MORE TRAINING','Rotation',90);
eval(tmp)
if isempty(bessere)
    ranking=1;
else
    ranking=bessere+1;
end
i=85;
d=3;

text(1,i,'Brain-Computer Interface Experiment: Demo','FontSize',18,'FontWeight','bold');
i=i-d-d;
text(1,i,'An Electroencephalogram-based Brain-Computer Interface (EEG-based BCI) provides a new communication','FontSize',10);
i=i-d;
text(1,i,'channel between the human brain and the computer. Patients who suffer from severe motor impairments');
i=i-d;
text(1,i,'(e.g. late stage of Amyotrophic Lateral Sclerosis (ALS), severe cerebral palsy, head trauma and spinal');
i=i-d;
text(1,i,'injuries) use such a BCI system as an alternative form of communication controlled by mental activity.');
i=i-d-d;
text(1,i,['Number of trials used: ',num2str(Numtrials)]);
 
i=i-d;
% text(1,i,['BCI Contest Ranking: ',int2str(Rranking)]);
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

text(1,i,'Realized with g.tec Highspeed and g.BSanalyze.','FontSize',10,'FontWeight','bold');

b=axes; 
set(b,'Position', [0.1300+0.4    0.2500    0.7750/2.05    0.8150/2.05]);
[A,B]=imread('paradigm.gif');

image(A,'Parent',b);
colormap(B)
axis off

clear P_C_copy;
disp(['Created classfier saved as: ',FileName]);
