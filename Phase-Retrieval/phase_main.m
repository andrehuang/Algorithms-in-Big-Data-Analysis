clear all;
rng(10);
n = 40;    % length of the true signal
s = 5;     % number of non-zero entries in the sparse true signal
m = 200;   % number of measurements

problemType = 'phaseless';      % phaseless measurements
measurementType = 'cplxgauss';  % (Complex) Random Gaussian measurements

TrueSignal = zeros(n,1);
TrueSignal(randperm(n,s) ) = (rand(s,1) - 0.5) + ...
                1i*(rand(s,1) - 0.5);

[measurements, measurementMat] = generateMeasurements(...
    problemType, measurementType, TrueSignal, m);

A = measurementMat;
b = measurements;
X = phase_cvx(A,b); % Use cvx to solve the sdp problem

% Recover the signal
[recoveredSig, eigenV] = eig(X);
recoveredSig = sqrt(eigenV(end))*recoveredSig(:,end);

% Correct for global phase factor
phaseFactor = exp( 1i* angle( (recoveredSig'*TrueSignal)/(TrueSignal'*TrueSignal) ) );
recoveredSig = recoveredSig*phaseFactor;

err_2 = norm(TrueSignal - recoveredSig)/norm(TrueSignal);
fprintf( '\nError in recovery (2-norm) is %3.3e \n', err_2);


