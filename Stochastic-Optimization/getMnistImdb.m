function imdb = getMnistImdb(dataDir)
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(dataDir, 'dir')
  mkdir(dataDir) ;
end

for i=1:4
  if ~exist(fullfile(dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, dataDir) ;
  end
end

f=fopen(fullfile(dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

% set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
%%% training and test data together
% data = single(reshape(cat(3, x1, x2),28,28,1,[]));  % catenate training data and test data together

% dataMean = mean(data(:,:,:,set == 1), 4);
% data = bsxfun(@minus, data, dataMean) ;

%%% only training data
data = single(reshape(x1,28,28,1,[]));
testdata = single(reshape(x2,28,28,1,[]));

imdb.images.data = data;
imdb.images.test = testdata;
imdb.images.labels = y1;
imdb.images.testlabels =y2;
