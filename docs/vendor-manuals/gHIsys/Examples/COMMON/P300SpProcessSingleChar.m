function [sys,x0,str,ts] = P300SpProcessSingleChar(t,x,u,flag,windTime,runmax)
%P300SpProcessSingleChar  Single-Character-Flash Signal Processing
%
%   The Signal Processing Block has the following tasks to perform:
%       - detect an input at the control line ID-Flash
%       - load the EEG-Data into the correct Buffer
%       - when the Run-Number maximum has reached the S-function has to
%         activate the STOP control line
%       - the function has to detect the letter with the maximum value in
%         the appropriate EEG-Buffer
%       - set the control line ID-RESULT with the number of the Solution
%         Letter
%
%   With double-click on the Signal Processing Block in the Simulink model,
%   you can set the following Options:
%
%   1. Buffer Window Time[ms] - The time how long the EEG-data of every
%                               letter should be saved
%
%   2. Maximum Number of Runs
%
% Author: Bernhard Grosswindhager
% Last Modified: 02.02.2011 by Maresch
% 2011 g.tec medical engineering GmbH

global userData

switch flag
    case 2 % Update of discrete states
        %----------------------------------------------------------
        % Save the EEG-Data in the AppendBuffer
        % This buffer is used for the BaseLineCorrection. It will be
        % appended in front of the EEG-Buffer of the Letter.
        %----------------------------------------------------------
        if (userData.buffIndex <= userData.appendBuffer.size)
            userData.appendBuffer.data(userData.buffIndex) = u(1);
            userData.buffIndex=userData.buffIndex+1;
        else
            userData.buffIndex = 1;
        end
        tLoadBuf = userData.tLoadBuf;
        if (userData.runnumber <= userData.runmax)
            if (u(2) ~= 0)
                input = u(2);
                userData.k = userData.k+1;
                % There is no buffer with the appropriate number!!
                if ~any(userData.arrFlashNum == input)
                    userData.i = userData.i + 1;
                    userData.arrFlashNum(userData.i) = input; % Save all inputs in arrays
                    size = userData.buffSize;
                    userData.sumBuffer(input).size = size;
                    userData.sumBuffer(input).data = zeros(1,size);
                    userData.sumBuffer(input).pos = input;
                    userData.runBuffer(input).size = size;
                    userData.runBuffer(input).data = zeros(1,size);
                end
                for i=1:numel(userData.flashIndex)
                    if (userData.flashIndex(i) == 0) % New Buffer has to be filled
                        userData.flashIndex(i) = input;
                        break;
                    end
                end
            end
            for elem=1:numel(userData.flashIndex)  % How many Buffers have to be filled parallel
                % > 1 if the Buffer has to be filled
                if ((userData.flashIndex(elem) > 0) && (userData.runnumber <= userData.runmax))
                    if userData.newrun(elem)
                        userData.newrun(elem)=false;
                        userData.tStart(elem) = t; % Set the new starttime
                    end
                    if (t > userData.tStart(elem))
                        if (userData.sample(elem) == 1) % First Sample
                            %-------------------------------------
                            % Append the 100ms Running-Buffer in
                            % front of the EEG-Buffer
                            %-------------------------------------
                            size1 = userData.appendBuffer.size;
                            size2 = userData.buffSize;
                            userData.runBuffer(userData.flashIndex(elem)).data(1:size1) = userData.appendBuffer.data(:);
                            userData.sample(elem) = size1+1;
                            if ((userData.k > numel(userData.runBuffer)) && (userData.runnumber == userData.runmax))
                                userData.output(1) = 1; % STOP
                            end
                        end
                        if (userData.k <= numel(userData.runBuffer)+1)
                            %----------------------------------
                            % Load EEG-Data into the buffer
                            %----------------------------------
                            userData.runBuffer(userData.flashIndex(elem)).data(userData.sample(elem)) = u(1);
                            userData.sample(elem) = userData.sample(elem)+1;
                        end
                    end
                    if (t > userData.tStart(elem)+tLoadBuf)
                        if (userData.k > numel(userData.runBuffer)) % If run is finished
                            userData.runnumber = userData.runnumber+1;
                            userData.k = 1;
                        end
                        userData.sample(elem) = 1; % First sample
                        userData.newrun(elem)=true; % Load the new time into handles.tStart
                        %----------------------------------
                        % Call the baseCorr function
                        %----------------------------------
                        baseCorr(userData.flashIndex(elem));
                        %----------------------------------
                        % Add the basecorrected Buffer of
                        % the last Run to the others
                        %----------------------------------
                        userData.sumBuffer(userData.flashIndex(elem)).data = ...
                            userData.sumBuffer(userData.flashIndex(elem)).data ...
                            + userData.runBuffer(userData.flashIndex(elem)).data;
                        
                        userData.runBuffer(userData.flashIndex(elem)).data(:) = 0;
                        userData.flashIndex(elem)=0;
                        if ~any(userData.flashIndex) % If all buffers are filled ready -> STOP
                            userData.loadBuffReady = true;
                            userData.runnumber = userData.runnumber+1;
                        end
                    end
                end
            end
        else
            if (userData.loadBuffReady == true) % Is set if all buffers are filled ready
                numelem = userData.runBuffer(1).size;
                for i=1:numel(userData.runBuffer)
                    %-----------------------------------------
                    % Calculate the average of the buffer
                    % to minimize noise!!!
                    %-----------------------------------------
                    userData.runBuffer(i).data = userData.sumBuffer(i).data;
                    userData.runBuffer(i).data(:) = userData.runBuffer(i).data(:)/userData.runmax;
                    userData.runBuffer(i).data(numelem) = 0;
                end
                solIndex = 0;
                firstSample = floor(0.3/(1/64)); % 300ms-100ms = 200ms after Flash
                lastSample = ceil(0.6/(1/64));   % 600ms-100ms = 500ms after Flash
                compArray=(firstSample : lastSample);
                maximum = 0;
                for i=1:numel(userData.runBuffer) % Calcualte the maximum of each buffer
                    maximum(i) = max(userData.runBuffer(i).data(compArray));
                end
                [temp index] = sort(maximum,'descend');
                solIndex = userData.sumBuffer(index(1)).pos;
                userData.output(2) = solIndex;
                
                %--------------------------------
                % Start new trial
                %--------------------------------
                userData.newtrial = true;
                userData.waitNextTrial = false;
                output = userData.output;
                %-----------------------------
                % Call the newInit function
                %-----------------------------
                newInit(windTime);
                userData.run = true;
                userData.output = output;
            else
                userData.runnumber = userData.runnumber - 1; % Cause not all buffers are loaded yet
            end
        end
        sys=[];
    case 3 % Calculates the outputs of the S-function
        sys = userData.output; % STOP + Solution ID
        userData.output = [0 0];
    case 0 % Initialization
        sizes=simsizes;
        sizes.NumContStates  = 0;
        sizes.NumDiscStates  = 0;
        sizes.NumOutputs     = 2;
        sizes.NumInputs      = -1;   % Dynamically sized
        sizes.DirFeedthrough = 0;    % Has no direct feedthrough
        sizes.NumSampleTimes = 1;
        
        sys=simsizes(sizes);
        
        x0  = [];
        str = [];
        ts  = [-1 0];   % Inherited sample time run at the same rate
                        % as the block to which it is connected
        
        %---------------------------------------
        % Save the parameters also in userData
        %---------------------------------------
        userData.windTime = windTime;
        userData.runmax = runmax; % Maximum number of runs - character flashes
        
        %-----------------------------
        % Call the newInit function
        %-----------------------------
        newInit(windTime);
end

%===============================================
% baseCorr function
% Baseline Correction: Correct for reference interval from 1 sample to 5 samples
%-----------------------------------------------
function baseCorr(index)

global userData

mean20samp = 0;
corrSample = 5;

mean5samp = mean(userData.runBuffer(index).data(1:corrSample));
mean5samp = -(mean5samp);
%----------------------------------------------------
% Add the average of the 5 samples to the EEG-Buffer
%----------------------------------------------------
userData.runBuffer(index).data=userData.runBuffer(index).data+mean5samp;

%===============================================
% newInit function
% Run this function when you start a new 'translation'
%-----------------------------------------------
function newInit(windTime)

global userData

%--------------------------------------------------------------------
% Create the 100ms buffer to buffer the last the 100ms of the EEG-data
% This buffer is used for calculating the BaseLine Correction!!
%--------------------------------------------------------------------
userData.appendBuffer = struct('numall',0); % Clear the Buffer
appendBuffTime = 0.1; % [s]
size = ceil((appendBuffTime)/(1/64)); % 100ms assumed (windTime*0.001)
userData.appendBuffer.data = zeros(1,size);
userData.appendBuffer.size = size;

%--------------------------------------------------------------------
% Clear the buffers which save the BCI-signals
% Sample Time: 1/64 s
%--------------------------------------------------------------------
userData.sumBuffer = struct(); % Clear sumBuffer structure
userData.runBuffer = struct(); % Clear runBuffer structure

userData.buffSize = ceil((windTime*0.001)/(1/64))+1; % e.g. 800ms: 53 elements

%-------------------------------------------------------------
% Intialize counting variables, boolean variables and constants
%-------------------------------------------------------------
userData.i = 0; % Counting variable - count which buffer has to be filled
userData.k = 1; % Counting variable
userData.buffIndex = 1; % Counting variable used for the 100ms-Buffer

numBuffpara = 6; % How many Buffers have to be filled parallel
% Holds the Flash Number of the parallel filled Buffers
userData.flashIndex = zeros(numBuffpara);
userData.sample = ones(numBuffpara); % Count the samples
userData.newrun = ones(numBuffpara);
userData.runnumber = 1; % Holds the number of the actual run

userData.arrFlashNum = 0; % Holds the Flash Numbers (input from Paradigm)

userData.newtrial = false;
% Only the time after the Flash!
userData.tLoadBuf = (windTime*0.001) - appendBuffTime;
userData.loadBuffReady = false;

userData.output = [0 0]; % Output variables
