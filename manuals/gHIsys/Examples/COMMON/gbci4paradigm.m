function [sys,x0,str,ts] = gbci4paradigm(t,~,u,flag,paradigm,runnumber)

% Christoph Guger, 10.10.2002
% 
% Last Modified: Mar 22.12.2010 -> model path detection

global fig

DelayBegin=7;
% TrialLength=8;
% RandomInterval=[0.5 2.5];
% Repetitions=40;

ret_sys = find_system(gcs,'RegExp','on','Name','BCI System');
cur_sys = [ret_sys{1} '/Gain'];

if (flag == 2)
    if (t < DelayBegin)
        return
    elseif t==DelayBegin
        set_param(cur_sys,'Gain','0');
    end
    if (any(get(0,'Children')==fig))
        if strcmp(get(fig,'Name'),'BCI Paradigm'),
            set(0,'currentfigure',fig);
            handles=get(gca,'UserData');
            if (handles.part7 == 0)
                if (handles.newtrial == 1)
                    handles.i=handles.i+1;
                    handles.starttime=t;
                    handles.newtrial=0;
                    % Set cross visible on
                    set(handles.line1,'Visible','on');
                    set(handles.line2,'Visible','on');
                    %drawnow;
                end
                if ((t > handles.starttime+(256/128)) && (handles.part1 == 0))
                    for ii=1:10
                        beep;
                    end
                    set_param(cur_sys,'Gain','1');
                    handles.part1=1;
                end
                if ((t > handles.starttime+(320/128)) && (handles.part4 == 0))
                    set_param(cur_sys,'Gain','0');
                    handles.part4=1;
                end
                if (t > handles.starttime+(384/128))&&(handles.part5 == 0)
                    % Show arrow
                    i=handles.i;
                    if (handles.pfeilrichtung(i) == 1)
                        set(handles.line8,'Visible','on');
                        set(handles.line5,'Visible','on');
                        set(handles.line6,'Visible','on');
                    else
                        set(handles.line7,'Visible','on');
                        set(handles.line3,'Visible','on');
                        set(handles.line4,'Visible','on');
                    end
                    %drawnow;
                    handles.part5=1;
                end
                if ((t > handles.starttime+(544/128)) && (handles.part2 == 0))
                    set(handles.line7,'Visible','off');
                    set(handles.line8,'Visible','off');
                    set(handles.line3,'Visible','off');
                    set(handles.line4,'Visible','off');
                    set(handles.line5,'Visible','off');
                    set(handles.line6,'Visible','off');
                    if (paradigm == 2)
                        set(handles.line2,'Visible','off');
                    else
                        set(handles.line2,'Visible','off');
                    end
                    %drawnow;
                    handles.part2=1;
                end
                if ((t > handles.starttime+(544/128)) && (handles.part3 == 0))
                    if (paradigm==2)
                        set(handles.bargraph,'Xdata',[0 u],'Ydata',[0 0],'Visible','on','Color','blue');
                        %drawnow;
                    else
                        set(handles.line2,'Visible','on','Color','blue');
                    end
                end
                if ((t > handles.starttime+(1000/128)) && (handles.part6 == 0))
                    if (paradigm == 2)
                        set(handles.bargraph,'Xdata',[0 0],'Ydata',[0 0],'Visible','off');
                    else
                        set(handles.line1,'Visible','off');
                        set(handles.line2,'Visible','off');
                    end
                    handles.part3=1;
                    handles.part6=1;
                    %drawnow;
                end
                if (t > handles.starttime+1100/128)
                    handles.newtrial=1;
                    handles.part1=0;
                    handles.part2=0;
                    handles.part3=0;
                    handles.part4=0;
                    handles.part5=0;
                    handles.part6=0;
                    if (handles.i == 40)
                        handles.part7=1;
                    end
                end
                set(gca,'UserData',handles);
            end
        end
    end
    sys=[];
elseif (flag == 0)
    %filename=[''];
    if (runnumber == 1)
        %session 1
        handles.pfeilrichtung=[0 0 0 1 0 0 1 1 1 0 1 1 1 0 0 1 0 0 1 0 1 0 ...
            0 0 1 1 0 1 1 0 1 1 1 0 0 1 0 0 1 1]; 
        %filename=[filename, '1'];
    elseif (runnumber == 2)
        %session 2
        handles.pfeilrichtung=[0 0 1 0 0 1 0 1 0 0 0 1 1 0 1 1 0 1 1 1 0 0 ...
            1 0 0 1 1 0 0 0 1 0 0 1 1 1 0 1 1 1]; 
        %filename=[filename, '2'];
    elseif (runnumber == 3)
        %session 3
        handles.pfeilrichtung=[1 0 0 1 0 0 1 0 1 0 0 0 0 0 1 0 0 1 1 1 0 1 ...
            1 0 1 1 0 1 1 0 1 1 1 0 0 1 0 0 1 1];
        %filename=[filename, '3'];
    elseif (runnumber == 4)
        %session 4
        handles.pfeilrichtung=[0 1 0 0 1 0 1 0 0 0 1 0 0 1 0 0 1 1 1 0 1 1 ...
            1 0 0 1 0 1 1 0 1 1 1 0 0 1 0 0 1 1];
        %filename=[filename, '4'];
    else
        fprintf('Wrong Runnumber');
    end
    
    % Initialize the figure for use with this simulation
    fig = animinit('BCI Paradigm');
    pos=get(0,'ScreenSize');
    set(fig,'HandleVisibility','on');
    set(fig,'Visible','off');
    set(fig,'Position',pos,'MenuBar','none');
    movegui(fig,'center');
    axis([-1 1 -1 1]);
    axis('off');
    hold on;
    
    sizes = simsizes;
    sizes.NumContStates  = 0;
    sizes.NumDiscStates  = 0;
    sizes.NumOutputs     = 0; % Dynamically sized
    sizes.NumInputs      = -1; % Dynamically sized
    sizes.DirFeedthrough = 0; % Has direct feedthrough
    sizes.NumSampleTimes = 1;

    sys = simsizes(sizes);
    str = [];
    x0  = [];
    ts  = [-1 0]; % Inherited sample time
    
    handles.line1=line([0 0], [0.5 -0.5],'Visible','off','EraseMode','Background');
    handles.line2=line([-0.5 0.5], [0 0],'Visible','off','EraseMode','Background');
    handles.line3=line([-0.5 -0.45],[0 0.025],'EraseMode','Background','Visible','off','LineWidth',3,'Color','r');
    handles.line4=line([-0.5 -0.45],[0 -0.025],'EraseMode','Background','Visible','off','LineWidth',3,'Color','r');
    handles.line5=line([0.5 0.45],[0 0.025],'EraseMode','Background','Visible','off','LineWidth',3,'Color','r');
    handles.line6=line([0.5 0.45],[0 -0.025],'EraseMode','Background','Visible','off','LineWidth',3,'Color','r');
    handles.line7=line([-0.5 0], [0 0], 'EraseMode','Background','Visible','off','LineWidth',3,'Color','r');
    handles.line8=line([0 0.5], [0 0], 'EraseMode','Background','Visible','off','LineWidth',3,'Color','r');
    handles.bargraph=plot([0 0],[0 0], 'LineWidth',5,'EraseMode','Background','Visible','off','Color','b');
    handles.newtrial=1;
    handles.i=0;
    handles.part1=0;
    handles.part2=0;
    handles.part3=0;
    handles.part4=0;
    handles.part5=0;
    handles.part6=0;
    handles.part7=0;
    
    set_param(cur_sys,'Gain',num2str(runnumber * -0.1));
    set(gca,'UserData',handles);
    set(fig,'Visible','on');
elseif (flag == 9)
    h=findobj('Name','BCI Paradigm');
    close(h);       
end
