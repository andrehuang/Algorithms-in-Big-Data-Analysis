% main test body
clear all;
batchSize = 300; % to stablize training, we set batchsize to be a litter larger than common practice in machine learning
max_iter = 10000;
lr = 1e-3;  
epsilon = 1e-10; % epsilon for adagrad? default
epsilon1 = 1e-8; % epsilon for adam, default
lambda = 10; % l1 regularizer
beta1 = 0.9;
beta2 = 0.999;
epsilon2 = 1e-5;  % stopping rule

dataDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'mnist') ;% directory
imdb = getMnistImdb(dataDir);  
Data = imdb.images;
X = Data.data;  % [28 28 1 60000]
X = reshape(X, [28*28, 60000]);  % [784, 1, 60000]
X1 = Data.test;
X1 = reshape(X1, [28*28, 10000]);

Y = Data.labels; % [1 60000]
Y1 = Data.testlabels; % [1 10000]
Y = mod(Y,2) == 0;  
Y = 2*(Y-0.5);  % label is [-1, 1]
Y1 = mod(Y1,2) == 0;  
Y1 = 2*(Y1-0.5);  % label is [-1, 1]

fprintf('Proximal AdaGrad Training \n')
w = randn([784, 1])/255;
g_sqr = zeros([784, 1]);
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
    idx = randi(60000, [batchSize ,1]);   
    x = X(:,idx); % [784, batchSize] 
    y = Y(:, idx); % [1 batchSize]
    opts = [lr, epsilon1, beta1, beta2, lambda];
    [loss1, w, g_sqr] = ProximalAda(@l1regfunction,x, y, w, g_sqr, opts);
    LOSS1(epoch) = loss1;
    epoch = epoch+1;
end
epoch1 = epoch-1

fprintf('ProximalAdam Training \n')
w = randn([784, 1])/255;
state = 0;
% for epoch = 1:max_iter
loss = 0;
loss1 = 1;
epoch = 1;
while norm(loss1 - loss, 2) > epsilon2
    if epoch == max_iter
        error_now = norm(loss1-loss,2)
        break
    end
    if epoch ~= 1
    loss = loss1;
    end
    idx = randi(60000, [batchSize ,1]);   
    x = X(:,idx); % [784, batchSize] 
    y = Y(:, idx); % [1 batchSize]
    opts = [lr/10, epsilon1, beta1, beta2, lambda];  % usually Adam needs smaller learning rate
    [loss1, w, state] = ProximalAdam(@l1regfunction,x, y, w, state, opts);
    LOSS2(epoch) = loss1;
    epoch = epoch +1;
end
epoch2 = epoch-1

fprintf('ProximalStabAda Training \n')
gamma = 1; % 
w = randn([784, 1])/255;
state = 0;
% for epoch = 1:max_iter
loss = 0;
loss1 = 1;
epoch = 1;
while norm(loss1 - loss, 2) > epsilon2
    if epoch == max_iter
        error_now = norm(loss1-loss,2)
        break
    end
    if epoch ~= 1
    loss = loss1;
    end
    idx = randi(60000, [batchSize ,1]);   
    x = X(:,idx); % [784, batchSize] 
    y = Y(:, idx); % [1 batchSize]
    opts = [lr*2, epsilon1, beta1, beta2, gamma, lambda]; % usually StabAda needs larger learning rate
    [loss1, w, state] = ProximalStabAda(@l1regfunction,x, y, w, state, opts);
    LOSS3(epoch) = loss1;
    epoch = epoch+1;
end
epoch3 = epoch-1

figure;
plot(1:epoch1, LOSS1, 1:epoch2, LOSS2, 1:epoch3, LOSS3, 'linewidth',1.2)
legend('ProximalAdaGrad', 'ProximalAdam', 'ProximalStabAda')
xlabel('iterations')
ylabel('training error')
% figure;
% plot(TestError1(:,1), TestError1(:,2), TestError2(:,1), TestError2(:,2), TestError3(:,1), TestError3(:,2),...
%     TestError4(:,1), TestError4(:,2),'linewidth',1.2)
% legend('Adagrad', 'Adam', 'StabAda', 'ProximalAda')
% xlabel('time')
% ylabel('test error')