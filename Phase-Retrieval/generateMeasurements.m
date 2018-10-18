function [measurements, measurementMat, noise] = generateMeasurements( ...
        problemType, measurementType, trueSignal, m, varargin )
% generateMeasurements Generate linear or magnitude measurements
%   Generate linear measurements 
%           b = M x + n,
%   or magnitude measurements 
%           b = |M x|^2 + n,
%               where   M \in C^{mxN} is a measurement matrix
%                       n \in R^m is measurement noise
%                       x \in C^N is the unknown signal
%                       b \in R^m is the measured signal
%
%   Inputs:
%       problemType     - type of problem to generate measurements for
%                           phaseless - generate magnitude measurements
%                           linear    - generate linear measurements
%       measurementType - type of measurements to generate; choices are
%                           randgauss - random Gaussian measurements
%                           cplxgauss - complex random Gaussian measurements
%                           bernoulli - Bernoulli measurements
%                           fourier   - Fourier measurements
%                           2stagerand- two-stage measurements for sparse 
%                                       phase retrieval (random Gaussian)
%       trueSignal      - the underlying true signal
%       m               - No. of measurements
%
%   Optional Argumens:
%       mCS             - For two-stage sparse phase retrieval problems,
%                         this is the CS problem dimension
%
%   Outputs:
%       measurements    - the generated measurements
%       measurementMat  - Matrix used to generate measurements
%                           For the two-stage sparse phase retrieval
%                           formulation, this is a structure containing the
%                           phase retrieval matrix, P, and the CS matrix C
%

% Length of true signal
n = length(trueSignal);

% Process optional arguments
if(nargin == 6)
    mCS = varargin{1};
end

% type of measurements
switch lower(problemType)
    % Linear measurements
    case 'linear'
        
        % Generate measurement matrix
        switch lower(measurementType)
            case 'randgauss'
                % Random Gaussian measurements
                measurementMat = randn(m,n);
                
            case 'cplxgauss'
                % Complex random Gaussian measurements
                measurementMat = randn(m,n) + 1i*randn(m,n);
                
            case 'bernoulli'
                % Bernoulli measurements
                measurementMat = sign(rand(m,n) - 0.5);
                
            case 'fourier'
                % Fourier measurements
                measurementMat = exp( -2i*pi*(0:m-1).'*(0:n-1)/n );
        end
        
        % Generate measurements
        measurements = measurementMat*trueSignal;
    % Phaseless measurements
    case 'phaseless'
        % Generate measurement matrix
        switch lower(measurementType)
            case 'cplxgauss'
                % Complex random Gaussian measurements
                measurementMat = randn(m,n) + 1i*randn(m,n);
                
                % Generate magnitude measurements
                measurements = abs( measurementMat*trueSignal ).^2;

            case '2stagerand'
                % Random (complex) Gaussian measurements for the 2-stage
                % sparse phase retrieval problem formulation
                C = randn(mCS, n) + 0i*randn(mCS, n);   % CS matrix
                P = randn(m, mCS) + 1i*randn(m, mCS);   % Phase Retrieval 
                                                        % matrix
                M = P*C;            % Composite measurement matrix
                
                % Store components of measurement matrix in a structure
                measurementMat.M = M;
                measurementMat.P = P;
                measurementMat.C = C;
                
                % Generate magnitude measurements
                measurements = abs( M*trueSignal ).^2;
        end
end
return