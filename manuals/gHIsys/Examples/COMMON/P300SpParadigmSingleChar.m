function [sys,x0,str,ts] = P300SpParadigmSingleChar(t,x,u,flag,mode,flashtime,darktime,nrchar)
% Single Character Flash Speller
%
% Version 1.0
% Last Modified: 21.08.2007 by Maresch
% 1999-2007 g.tec medical engineering GmbH

persistent fig parahandles

switch flag
    case 'New' % The new Game Button was pressed
        set(gcf,'UserData',parahandles); % Save the parahandles object in the figure's UserData
        newInit();
        parahandles = get(gcf,'UserData'); % Load the changes of the newInit function
        set(parahandles.outputText,'Visible','off');
        set(parahandles.flashFields,'Visible','off');
        set(parahandles.frame,'Visible','off'); % Don't show the background image
        set(parahandles.smily_well,'Visible','off');
        set(parahandles.smily_middle,'Visible','off');
        set(parahandles.smily_bad,'Visible','off');
        set(parahandles.startHndl,'Visible','on'); % Show the Start Button
        set(parahandles.runnumberText,'String','0');
        set(parahandles.copyText,'Visible','off'); % Set the Text with the spelling info invisible
        set(parahandles.spellText,'String','','Visible','off'); % Clear the Spell Text and set invisible
            
        if (parahandles.mode == 1) % Copy Spelling
            set(parahandles.copyEdit,'Visible','on');
            parahandles.copySpell = true;
            set(parahandles.copyText,'String','Please enter the Word for Copy Spelling', ...
                'Visible','on');
        end
        set(gcf,'UserData',parahandles); % Save the changes
        return; % Stop this step
    case 'UpdateCopySpell' % The CallbackFunction of the Buttons, where
                           % you can enter the SpellWord
            parahandles.copySpell = false; % Now you can press the start button
            trialmax = parahandles.trialmax; % Count how often you press a button
            trialmax = trialmax+1; % Afterwards this is the number of trials
            parahandles.trialmax = trialmax;
            for i=1:numel(parahandles.arrFlashIndex) % Which button was pressed
                if get(parahandles.copyEdit(i),'Value')
                    index = i;
                    break;
                end
            end
            copyString = get(parahandles.copyEdit(index),'String'); % The name of the pressed button
            parahandles.copyStrings{trialmax} = copyString; % Save the pressed buttons to a cell array
            if (trialmax == 1) % First time the program jumps into the Callback Routine
                oldcopyString = '';
                newcopyString = copyString;
            else
                oldcopyString = get(parahandles.copyText,'String');
                oldcopyString = [oldcopyString ' '];
                newcopyString = [oldcopyString copyString];
            end
            set(parahandles.copyText,'String',newcopyString); % Load the new copy Text into the Textfield
            set(gcf,'UserData',parahandles); % Save the changes
            return; % Stop this step
    case 'Start'
        parahandles.newtrial = true;
        parahandles.trialnumber = 1; % Holds the number of the actual trial
        parahandles.run = true; % Start the 'translation'
        if (parahandles.mode == 1) % Copy Spelling
            set(parahandles.copyEdit,'Visible','off');
        end
        if (parahandles.mode == 2)
            parahandles.trialmax = inf;
        end
        set(parahandles.copyText,'Visible','on');
        set(parahandles.spellText,'Visible','on');
        set(parahandles.startHndl,'Visible','off'); % Don't show the Start Button
        set(parahandles.frame,'Visible','on'); % Show the background image
        set(parahandles.flashFields,'Visible','on');
        set(gcf,'UserData',parahandles); % Save changes
        return; % Stop this step
    case 'Closefig'
        close(gcf); % Close figure window
        return; % Stop this step
    case 2 % Update of discrete states
        if any(get(0,'Children') == fig) % Is fig a 'Child' of the root object?
            if strcmp(get(fig,'Name'),'BCI P300 Matrix Speller - Single Character Flash'),
                set(0,'currentfigure',fig); % Set fig to the current figure
                parahandles=get(gcf,'UserData'); % Load the parahandles obejct from UserData
                stop = u(1);
                if (stop ~= 0)
                    parahandles.stop = stop;
                end
                if parahandles.run
                    if parahandles.newtrial % True...the program has to wait
                        parahandles.newtrial=false; % before starting the next trial
                        parahandles.waitNextTrial=true;
                        parahandles.starttimeTrial=t;
                    end
                    if ~parahandles.waitNextTrial % False...Trial not ready
                        tDarkLetter = parahandles.tDarkLetter;
                        tFlash = parahandles.tFlash;
                        if (parahandles.newrun && parahandles.stop ~= true)
                            parahandles.newrun = false;
                            parahandles.starttime = t; % Set the new starttime
                            parahandles.flashIndex = parahandles.randarr(parahandles.k);
                            parahandles.k = parahandles.k+1;
                        end
                        if (parahandles.stop ~= true)
                            if (t > parahandles.starttime)
                                if parahandles.draw % Highlight the object, but only once
                                    %-----------------------------
                                    % Call the setClearTrigger function
                                    %-----------------------------
                                    set(gcf,'UserData',parahandles); % Save the parahandles object
                                    setClearTrigger(t,parahandles.flashIndex); % Set Trigger
                                    parahandles = get(gcf,'UserData'); % Load the changes of the function
                                    parahandles.statTrigger = true;
                                    set(parahandles.flashFields(parahandles.flashIndex), ...
                                        'ForegroundColor','white');
                                    drawnow
                                    parahandles.output(1) = parahandles.flashIndex;
                                    parahandles.draw = false;
                                elseif parahandles.statTrigger
                                    %-----------------------------
                                    % Call the setClearTrigger function
                                    %-----------------------------
                                    set(gcf,'UserData',parahandles); % Save the parahandles object
                                    setClearTrigger(t,0); % Clear Trigger
                                    parahandles.statTrigger = false;
                                end
                            end
                        else
                            parahandles.newrun = false;
                        end                      
                        if (t > (parahandles.starttime + tFlash)) % Clear the FlashFields
                            if parahandles.clear
                                set(parahandles.flashFields,'ForegroundColor',(40/255)*[1 1 1]);
                                drawnow
                                parahandles.clear=false;
                                if (parahandles.k > numel(parahandles.arrFlashIndex))
                                    lastelement = parahandles.randarr(numel(parahandles.arrFlashIndex));
                                    parahandles.randarr(1) = lastelement;
                                    %-------------------------
                                    % Random Flash order
                                    %-------------------------
                                    while(parahandles.randarr(1) == lastelement)
                                        parahandles.randarr = randperm(numel(parahandles.arrFlashIndex));
                                    end
                                    set(parahandles.runnumberText,'String',parahandles.runnumber);
                                    parahandles.runnumber = parahandles.runnumber+1;
                                    parahandles.k = 1;
                                end
                            end
                        end
                        if (t > (parahandles.starttime + tFlash + tDarkLetter)) % The next Letter will flash on the screen
                            parahandles.draw=true;
                            parahandles.clear=true;
                            parahandles.newrun=true; % Load the new time into parahandles.starttime
                        end
                        if (parahandles.stop == true)
                            parahandles.newrun = false;
                            solIndex = u(2);
                            if (solIndex ~= 0)
                                newLetter = get(parahandles.flashFields(solIndex),'String');
                                oldspellText = get(parahandles.spellText,'String');
                                if ~isempty(oldspellText)
                                    oldspellText = [oldspellText ' '];
                                end
                                newspellText = [oldspellText newLetter];
                                set(parahandles.spellText,'String', ...
                                    newspellText);
                                if (mode == 1) % Copy Spelling
                                    if strcmp(newLetter,parahandles.copyStrings(parahandles.trialnumber))
                                        set(parahandles.outputText,'String','Translation successful!', ...
                                            'Visible','on');
                                        set(parahandles.spellText,'BackgroundColor','g');
                                        parahandles.correctTrials = parahandles.correctTrials + 1;
                                    else % Free Spelling
                                        set(parahandles.spellText,'BackgroundColor','r');
                                        parahandles.wrongTrials = parahandles.wrongTrials + 1;
                                        set(parahandles.outputText,'String','Wrong Character!', ...
                                            'Visible','on');
                                    end
                                end
                                if ((mode == 1) && (parahandles.trialnumber == parahandles.trialmax)) % CopySpelling
                                    parahandles.run = false; % Stop the 'translation'
                                    accuracy = round((parahandles.correctTrials/parahandles.trialmax)*100);
                                    if (accuracy <= 35)
                                        set(parahandles.smily_bad,'Visible','on');
                                    elseif (accuracy <= 65)
                                        set(parahandles.smily_middle,'Visible','on');
                                    elseif (accuracy > 65)
                                        set(parahandles.smily_well,'Visible','on');
                                    end
                                    set(parahandles.outputText, ...
                                        'String',sprintf('Correct: %d   Wrong: %d   Accuracy: %d%%', ...
                                        parahandles.correctTrials,parahandles.wrongTrials,accuracy), ...
                                        'Color', 'black', ...
                                        'FontSize', 0.04, ...
                                        'Visible','on');
                                end
                                %--------------------------------
                                % Start new trial
                                %--------------------------------
                                parahandles.stop = false;
                                parahandles.newtrial = true;
                                parahandles.trialnumber = parahandles.trialnumber+1;
                            end                      
                        end
                    else
                        if (t > (parahandles.starttimeTrial+parahandles.trialwaitTime))
                            set(parahandles.runnumberText,'String',1);
                            parahandles.waitNextTrial = false;
                            trialmax = parahandles.trialmax;
                            copyStrings = parahandles.copyStrings;
                            correctTrials = parahandles.correctTrials;
                            wrongTrials = parahandles.wrongTrials;
                            %-----------------------------
                            % Call the newInit function
                            %-----------------------------
                            set(gcf,'UserData',parahandles); % Save the parahandles object
                            newInit();
                            parahandles = get(gcf,'UserData'); % Load the changes of the function
                            set(parahandles.outputText,'Visible','off');
                            parahandles.run = true;
                            parahandles.trialmax = trialmax;
                            parahandles.copyStrings = copyStrings;
                            parahandles.correctTrials = correctTrials;
                            parahandles.wrongTrials = wrongTrials;
                        end
                    end
                end
                set(gcf,'UserData',parahandles); % Save changes to UserData of the current figure
            end
        end
        sys=[];
    case 3 % Calculates the outputs of the S-function
        if ishandle(fig) % If the figure still exists for example when the
                                         % close button was already pressed.
            parahandles=get(fig,'UserData'); % Load the parahandles obejct from UserData
            sys = parahandles.output;        % called at every sample
            parahandles.output = [0 0];      % Set only during one sample
            set(fig,'UserData',parahandles); % Save the parahandles object in the figure's UserData
        end
    case 0 % Initialization
        %------------------------------------
        % Clear the parahandles object and close
        % the old figure if it is still open
        %------------------------------------
        %clear parahandles;
        h=findobj('Name','BCI P300 Matrix Speller - Single Character Flash');
        close(h);
        %------------------------------------
        % Initialize the Figure
        %------------------------------------
        figure('Name','BCI P300 Matrix Speller - Single Character Flash', ...
            'NumberTitle','off');
        fig = findobj('Type','figure','Name','BCI P300 Matrix Speller - Single Character Flash');
        set(fig,'Visible','off','NumberTitle','off');
        pos=get(0,'ScreenSize'); % Get the size of the screen
        set(fig,'Position',pos,'MenuBar','none','Units','normalized');
        movegui(fig,'center'); % Move GUI figure to specified part of screen
        axesHndl = axes('Position',[0 0 1 1]);
        axis([-1 1 -1 1]); % Sets scaling for the x- and y-axes on the current plot
        axis('off'); % Turns off all axis labeling
        hold on; % Holds the current plot and all axis properties so that
                 % subsequent graphing commands add to the
                 % existing graph
        sizes=simsizes; % SIMSIZES...utility used to set S-function sizes
        sizes.NumContStates  = 0; % Number of continuous states
        sizes.NumDiscStates  = 0; % Number of discrete states
        sizes.NumOutputs     = 2; % Number of outputs
        sizes.NumInputs      = 2; % Number of inputs
        sizes.DirFeedthrough = 0; % Has direct feedthrough
        sizes.NumSampleTimes = 1; % Number of sample times
        sys=simsizes(sizes); % After initializing the structure above to fit the
                             % specifications of the S-function, SIMSIZES should be called
                             % again to convert the structure into a vector that can be 
                             % processed by Simulink. For example:
                             % sys = simsizes(sizes);
        x0  = [];
        str = [];
        ts  = [-1 0]; % Inherited sample time run at the same rate
                      % as the block to which it is connected 
        numrows = 6;  % How many Rows and Columns should have the Field
        numcols = 6;
        parahandles.numrows = numrows;
        parahandles.numcols = numcols;
        
        FWIDTH = 0.09; % Width of one Field
        FHEIGHT = 0.11; % Height of one Field
        
        xoffset = ((0.80)/2) - (numrows/2)*FWIDTH; % x-beginning of the field
        yoffset = (0.47) + (numcols/2-1)*FWIDTH;   % y-beginning of the field

        B = cell(numrows, numcols); % Cell array which holds the Letters or words
  
        % Select the 36 characters or 26 characters
        parahandles.nrchar=nrchar;
        if (parahandles.nrchar == 1)
            B = {'A','B','C','D','E','F'
                 'G','H','I','J','K','L'
                 'M','N','O','P','Q','R'
                 'S','T','U','V','W','X'
                 'Y','Z','1','2','3','4'
                 '5','6','7','8','9','_'};
        elseif (parahandles.nrchar == 2)
            B = {'A','B','C','D','E',''
                 'F','G','H','I','J','',
                 'K','L','M','N','O','',
                 'P','Q','R','S','T','',
                 'U','V','W','X','Y','',
                 'Z','','','','',''};  
        end
        
        %-----------------------------------------
        % Load the Smily Images into Workspace
        %-----------------------------------------
        smilexPos = 0.44;
        smileyPos = 0.86;
        smileWidth = 0.08;
        smileHeight = 0.11;
        smily_well = imread('smily_well','jpg');
        parahandles.smily_well = image( ...
            [smilexPos smilexPos+smileWidth],[smileyPos+smileHeight smileyPos], ...
            smily_well, ...
            'Visible','off');
        smily_middle = imread('smily_middle','jpg');
        parahandles.smily_middle = image( ...
            [smilexPos smilexPos+smileWidth],[smileyPos+smileHeight smileyPos], ...
            smily_middle, ...
            'Visible','off');
        smily_bad = imread('smily_bad','jpg');
        parahandles.smily_bad = image( ...
            [smilexPos smilexPos+smileWidth],[smileyPos+smileHeight smileyPos], ...
            smily_bad, ...
            'Visible','off');
        %---------------------------------
        % TextField for Copy or Free Spelling
        %---------------------------------
        parahandles.copyText = uicontrol('Style','text', ...
            'Units','normalized', ...
            'FontUnits','normalized', ...
            'FontName','Courier',...
            'FontWeight','bold',...
            'FontSize',0.6, ...
            'BackgroundColor',192/255*[1 1 1], ...
            'HorizontalAlignment','left', ...
            'Position', [xoffset, yoffset+FHEIGHT*1.8, FWIDTH*numcols, FHEIGHT*0.4], ...
            'String', '', ...
            'Visible','off');
        %---------------------------------
        % TextField, where the solution will be displayed
        %---------------------------------
        parahandles.spellText = uicontrol('Style','text', ...
            'Units','normalized', ...
            'FontUnits','normalized', ...
            'FontName','Courier',...
            'FontWeight','bold',...
            'FontSize',0.7, ...
            'BackgroundColor', 192/255*[1 1 1], ...
            'HorizontalAlignment','left', ...
            'Position', [xoffset, yoffset+FHEIGHT*1.4, FWIDTH*numcols, FHEIGHT*0.4], ...
            'String', '', ...
            'Visible','off');
        %--------------------------------
        % the black background image
        %--------------------------------
        parahandles.frame = uicontrol('Style','frame',...
            'Units','normalized',...
            'Position', [xoffset,yoffset-((numrows-1)*FHEIGHT), ...
            FWIDTH*numcols,FHEIGHT*(numrows+0.4)],...
            'BackgroundColor',[0 0 0], ...
            'Visible','off');
        
        parahandles.arrFlashIndex = zeros(1);
        fontSize = 0.35;
        k = 1;
        i = 0;
        %-----------------------------------------------
        %The Textfields with the Letters or the words
        %  These Fields will flash on the screen
        %-----------------------------------------------
        for m = 1:numrows
            for n = 1:numcols
                i=i+1;
                if ~isempty(B{m,n}) % Is the matrix not empty --> create a flash Field
                    parahandles.flashFields(k) = uicontrol('Style','text', ...
                        'Units','normalized', ...
                        'FontUnits','normalized', ...
                        'FontSize',fontSize, ...
                        'FontWeight','bold', ...
                        'BackgroundColor','black', ...
                        'ForegroundColor',(40/255)*[1 1 1], ...
                        'Position', [xoffset+((n-1)*FWIDTH),yoffset-((m-1)*FHEIGHT),FWIDTH,FHEIGHT], ...
                        'String',B{m,n}, ...
                        'Visible','off');
                    parahandles.arrFlashIndex(k) = k;
                    if (mode == 1) % Copy Spelling
                        labelString = B{m,n};
                        callback='P300SpParadigmSingleChar([],[],[],''UpdateCopySpell'',[])';
                        btnPos = [xoffset+((n-1)*FWIDTH), ...
                            yoffset-((m-2.7)*FHEIGHT/2),FWIDTH,FHEIGHT/2];
                        parahandles.copyEdit(k)= uicontrol( 'Style','pushbutton', ...
                            'Parent',fig, ...
                            'Units','normalized', ...
                            'FontSize',10, ...
                            'Position',btnPos, ...
                            'String',labelString, ...
                            'Callback',callback, ...
                            'BackgroundColor',(200/255)*[1 1 1]);
                    end
                    k = k+1;
                end
            end
        end
       
        %-------------------------------
        % Output TextField
        %-------------------------------
        parahandles.outputText = text(-0.2,0.91,'Translation successful!', ...
            'FontUnits','normalized',...
            'HorizontalAlignment','center', ...
            'FontSize',0.04, ...
            'FontAngle','italic', ...
            'FontName','Times New Roman',...
            'FontWeight','bold', ...
            'Color', 'black', ...
            'Visible','off');
        set(parahandles.outputText,'String','Yes-No Decision detected');
        
        %-----------------------------
        % Save the parameters also in the parahandles structure
        %-----------------------------
        parahandles.mode = mode;
        parahandles.flashtime = flashtime/1000;
        parahandles.darktime = darktime/1000;

        %-----------------------------
        % Call the newInit function
        %-----------------------------
        set(gcf,'UserData',parahandles); % Save the parahandles object in the figure's UserData
        newInit();
        parahandles = get(gcf,'UserData'); % Load the changes of the newInit function
        
        if (mode == 1) %Copy Spelling
            set(parahandles.copyText,'Visible','on');
            set(parahandles.spellText,'Visible','on');
            parahandles.copySpell = true;
            set(parahandles.copyText,'String','Please enter the Word for Copy Spelling');
            parahandles.copyStrings=cell(1); %this cellArray is used in the UpdateSpell
        else % Free Spelling
            parahandles.copySpell = false;
        end
        
        %======================================================================
        % Buttons
        %======================================================================
        % Information for all buttons
        yInitPos=0.90;
        top=0.95;
        left=0.80;
        bottom=0.05;
        btnWidth=0.15;
        btnHeight=0.10;
        % Space between the button and the next command's label
        space=0.04;
        %====================================
        % Create the UIPANEL
        panBorder=0.02;
        yPos=0.05-panBorder;
        panPos=[left-panBorder yPos btnWidth+2*panBorder 0.9+2*panBorder];
        parahandles.uipan = uipanel( ...
            'Parent',fig, ... 
            'Units','normalized', ... 
            'Position',panPos, ...
            'BackgroundColor',[0.50 0.50 0.50], ...
            'Visible','on');
        
        %====================================
        % Debugging Information (runnumber & 2 axes)
        parahandles.runnumberText = uicontrol('Style','text', ...
            'Parent',parahandles.uipan, ...
            'Units','normalized', ...
            'FontUnits','normalized', ...
            'FontSize',0.6, ...
            'Position', [0.75, 0.71, 0.2, 0.07], ...
            'BackgroundColor',[0.50 0.50 0.50], ...
            'String', 1, ...
            'Visible','off');

        %====================================
        % The START Button
        if (mode == 1) % Copy Spelling
            yPos = 0.4;       
        else % Free Spelling
            yPos = 0.6;
        end
        xPos = 0.325;
        labelStr = 'START';
        callbackStr='P300SpParadigmSingleChar([],[],[],''Start'',[])';
        
        % Generic Button Information
        btnPos=[xPos yPos-btnHeight btnWidth btnHeight];  
        parahandles.startHndl = uicontrol( ...
            'Style','pushbutton', ...
            'FontUnits','normalized', ...
            'FontSize',0.2, ...
            'FontWeight','bold', ...
            'Units','normalized', ...
            'Position',btnPos, ...
            'String',labelStr, ...
            'TooltipString','Starts the Translation', ...
            'Callback',callbackStr );
        
        %====================================
        % The NEW GAME button
        btnNumber=1;
        yPos=top-(btnNumber-1)*(btnHeight+space);
        labelStr='New Run';
        callbackStr='P300SpParadigmSingleChar([],[],[],''New'',[])';
        
        % Generic button information
        btnPos=[left yPos-btnHeight btnWidth btnHeight];
        uicontrol( ...
            'Style','pushbutton', ...
            'Units','normalized', ...
            'Position',btnPos, ...
            'String',labelStr, ...
            'TooltipString','Starts a new Translation', ...
            'Callback',callbackStr );
        
        %====================================
        % The CLOSE button
        labelStr='Close';
        callbackStr='P300SpParadigmSingleChar([],[],[],''Closefig'',[])';
        uicontrol( ...
            'Style','pushbutton', ...
            'Parent', fig, ...
            'Units','normalized', ...
            'Position',[left bottom btnWidth btnHeight], ...
            'String',labelStr, ...
            'TooltipString','Closes the P300 Speller Window', ...
            'Callback',callbackStr);        
        %====================================
        
        set(0,'currentfigure',fig); % After drawing in the other axes set
                                    % fig as the current figure
        set(gcf,'UserData',parahandles); % Save changes into UserData of the curr. figure
        set(fig,'Visible','on'); % Set figure visible

    case 9 % End of simulation tasks
        h=findobj('Name','BCI P300 Matrix Speller - Single Character Flash');
        close(h);
end

%------------------------------------------------------------
function newInit()
parahandles = get(gcf,'UserData'); % Load parahandles structure from UserData

%----------------------
% Random Flash order
%----------------------
parahandles.randarr = randperm(numel(parahandles.arrFlashIndex)); 

%-------------------------------------------------------------
% Set colors of uicontrols
%-------------------------------------------------------------
set(parahandles.flashFields,'ForegroundColor',(40/255)*[1 1 1]); % All Textfields dark
set(parahandles.spellText,'BackgroundColor', 192/255*[1 1 1]); % Original color

%-------------------------------------------------------------
% Intialize counting variables, boolean variables and constants
%-------------------------------------------------------------
parahandles.run = false;    % With the Start Button you can start the 'translation'
parahandles.stop = false;   % When the Signal Processing Block sends the STOP-Bit
                        % This variable will be set.
parahandles.flashIndex = 0; % The number of the currently flashing FlashField
parahandles.k = 1;          % Counting variable
parahandles.copyStrings=cell(1); % This cellArray is used in the UpdateCopySpell
                             % Callback-Fctn

parahandles.statTrigger = false; % True if the Trigger should be set
parahandles.draw = true;  % True if a Field could flash
parahandles.clear = true; % True if a Field could be cleared

parahandles.newrun = true; % New run will start
parahandles.runnumber = 1; % Holds the number of the actual run

parahandles.trialmax = 0; % Maximum number of trials, Copy Spelling trialmax is finite
                      % will be set in the 'UpdateCopySpell' Callback-fctn
parahandles.waitNextTrial = false;
parahandles.newtrial = false;
parahandles.trialwaitTime = 3; % [s] time before the next trial starts
parahandles.correctTrials = 0; % Holds the number of correct Trials
parahandles.wrongTrials = 0; % Holds the number of wrong Trials

parahandles.output = [0 0]; % Holds the output variable (sys), [id target stop]

parahandles.tDarkLetter = floor(parahandles.darktime/(1/64))*(1/64); % The time how long no letter
                                                             % should flash on the screen
parahandles.tFlash = floor(parahandles.flashtime/(1/64))*(1/64); % The FlashTime

set(gcf,'UserData',parahandles); % To save the changes of the newInit function

%============================================
% setClearTrigger function
%--------------------------------------------
function setClearTrigger(time,flashnum) % flashnum...parahandles.flashIndex
parahandles = get(gcf,'UserData'); % Load parahandles structure from UserData

if (flashnum ~= 0)
    %-------------------------------------------------
    % Set Trigger(Gain2), if the letter,on which you
    % have to look, flashes on the screen.
    %-------------------------------------------------
    if (parahandles.mode == 1) % CopySpelling
        flashLetter = get(parahandles.flashFields(flashnum),'String');
        if strcmp(flashLetter,parahandles.copyStrings(parahandles.trialnumber))
            parahandles.output(2)=1;
        end
    end
end
parahandles.output(1)=flashnum;
set(gcf,'UserData',parahandles); % To save the changes of the setTrigger function
