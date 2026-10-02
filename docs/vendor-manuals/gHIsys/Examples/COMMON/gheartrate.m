function [sys,x0,str,ts] = gheartrate(t,x,u,flag)

%Christoph Guger, 20.8.2007

global fig

if (flag == 2)
    if any(get(0,'Children') == fig)
        if strcmp(get(fig,'Name'),'Heart Rate')
            set(0,'currentfigure',fig);
            handles=get(gca,'UserData');        
            if (u(1) > 0.5)
                handles.time=t-handles.time1;
                handles.time1=t;
                HR=60/handles.time;
                if ((HR>handles.HR1*1.5) || (HR<handles.HR1*0.5))
                    HR=handles.HR1;
                end
                % Error detection algorithm
                set(handles.text1,'String',num2str(HR));                
                handles.HR1=HR;
                tmp=[handles.buffer HR];
                tmp(1)=[];
                handles.buffer=tmp;
                tmp=60./tmp;
                tmp(find(isnan([0 NaN 1])==1))=1;
                handles.RR=tmp*1000; % Convert to ms
            end
            if (u(2) > 0.5)
                handles.time=t-handles.time2;
                handles.time2=t;
                Resp=60/handles.time;
                handles.Resp1=Resp;
            end
            set(gca,'UserData',handles);
        end
    end
    sys=[];
elseif (flag == 3)
    handles=get(gca,'UserData');
    y=handles.HR1;
    meanHR=mean(handles.buffer);
    minHR=min(handles.buffer);
    maxHR=max(handles.buffer);
    stdHR=std(handles.buffer);
    
    diffRRms=diff(handles.RR);
    RMSSD=sqrt(mean((diffRRms.^2)));
    Resp=handles.Resp1;
    sys=[y meanHR minHR maxHR RMSSD Resp;];
elseif (flag == 0)
    % Initialize the figure for use with this simulation
    fig = animinit('Heart Rate');
    set(fig,'HandleVisibility','on')
    pos=get(0,'ScreenSize');
    set(fig,'Position',pos/5,'MenuBar','none');
    movegui(fig,'onscreen');
    axis([-1 1 -1 1]);
    axis('off');
    hold on;
    set(fig,'Visible','on');
    
    sizes = simsizes;
    sizes.NumContStates  = 0;
    sizes.NumDiscStates  = 0;
    sizes.NumOutputs     = 6; % Dynamically sized
    sizes.NumInputs      = -1; % Dynamically sized
    sizes.DirFeedthrough = 0; % Has direct feedthrough
    sizes.NumSampleTimes = 1;
    
    sys = simsizes(sizes);
    str = [];
    x0  = [];
    ts  = [-1 0];   % Inherited sample time
    
    handles.text1=text(0,0,'HR');
    handles.time1=0; % Time for ECG
    handles.time2=0; % Time for Respiration
    handles.time=0;
    handles.HR1=80;
    
    handles.Resp1=8;
    
    handles.buffer=ones(1,10)*60;
    tmp=rand(10);
    handles.RR=tmp(1,:)*1000;
    set(gca,'UserData',handles);
elseif (flag == 9)
    h=findobj('Name','Heart Rate');
    close(h);
end
