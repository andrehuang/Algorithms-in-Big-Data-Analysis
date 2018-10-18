% Predicting forest cover type from cartographic variables only
clear all;
batchSize = 1000;% to stablize training, we set batchsize to be a litter larger than common practice in machine learning
max_iter = 500;
lr = 1e-3;
epsilon = 1e-10; % epsilon for adagrad
epsilon1 = 1e-8; % epsilon for adam
lambda = 10; % l1 regularizer
beta1 = 0.9;
beta2 = 0.999;
epsilon2 = 1e-4; %stop rule

dataDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'covertype') ;
Data = getCovertypeImdb(dataDir);
% X = Data.data;  % [54 581012]
Y = Data.labels; % [1 581012] 

% unique(Y) % 1  2  3  4  5  6  7
% Y = dense_to_one_hot(Y); % [y, 581012]
X1 = Data.data(:, 1:400000);  % training data
X2 = Data.data(:, 400001:581012);  % test data
Y1 = Data.labels(:, 1:400000);  % training labels
Y2 = Data.labels(:, 400001:581012);   % test label
Y1 = mod(Y1,2) == 0;  
Y1 = 2*(Y1-0.5);  % label is [-1, 1]
Y2 = mod(Y2,2) == 0;  
Y2 = 2*(Y2-0.5);  % label is [-1, 1]

fprintf('ProximalAdaGrad Training \n')
% ----- it's very important to choose w properly
w = rand([54, 1])/100;  % VERY IMPORTANT
g_sqr = zeros([54, 1]);
epoch = 1;
loss = 0;
loss1 = 1;
while norm(loss1 - loss, 2) > epsilon2
    if epoch == max_iter
        error_now = norm(loss1-loss,2)
        break
    end
    if epoch ~= 1
    loss = loss1;
    end
    idx = randi(400000, [batchSize ,1]);   
    x = X1(:,idx);
    y = Y1(:, idx); 
    opts = [lr, epsilon, lambda];
    [loss1, w, g_sqr] = ProximalAda(@l1regfunction, x, y, w, g_sqr, opts);
    LOSS1(epoch) = loss1;
    epoch = epoch+1;
end
epoch1 = epoch -1

fprintf('ProximalAdam Training \n')
w = rand([54, 1])/100;
state = 0;
epoch = 1;
loss = 0;
loss1 = 1;
while norm(loss1 - loss, 2) > epsilon2
    if epoch == max_iter
        error_now = norm(loss1-loss,2)
        break
    end
    if epoch ~= 1
    loss = loss1;
    end
    idx = randi(400000, [batchSize ,1]);   
    x = X1(:,idx); 
    y = Y1(:, idx); 
    opts = [lr, epsilon1, beta1, beta2, lambda];
    [loss1, w, state] = ProximalAdam(@l1regfunction,x, y, w, state, opts);
    LOSS2(epoch) = loss1;
    epoch = epoch +1;
end
epoch2 = epoch -1

fprintf('ProximalStabAda Training \n')
gamma = 1;
w = rand([54, 1])/100;
state = 0;
tic;
epoch = 1;
loss = 0;
loss1 = 1;
while norm(loss1 - loss, 2) > epsilon2
    if epoch == max_iter
        error_now = norm(loss1-loss,2)
        break
    end
    if epoch ~= 1
    loss = loss1;
    end
    idx = randi(400000, [batchSize ,1]);   
    x = X1(:,idx); % [784, batchSize] 
    y = Y1(:, idx); % [1 batchSize]
    opts = [4*lr, epsilon1, beta1, beta2, gamma, lambda];
    [loss1, w, state] = ProximalStabAda(@l1regfunction,x, y, w, state, opts);
    LOSS3(epoch) = loss1;
    epoch = epoch + 1;
end
epoch3 = epoch -1

plot(1:epoch1, LOSS1, 1:epoch2, LOSS2, 1:epoch3, LOSS3, 'linewidth',1.2)
legend('ProximalAdagrad', 'ProximalAdam', 'ProximalStabAda')
xlabel('iterations')
ylabel('training loss')