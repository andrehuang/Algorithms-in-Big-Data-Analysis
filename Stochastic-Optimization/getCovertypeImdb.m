function imdb = getCovertypeImdb(dataDir)
% Preapre the imdb structure, returns image data with mean image subtracted
% file = fullfile(dataDir, 'covtype.data');
datadir = fullfile(dataDir, 'covtype.data');

if ~exist(dataDir, 'dir')
  mkdir(dataDir) ;
end


if ~exist(datadir, 'file')
    url = sprintf('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz') ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, dataDir) ;
end

% datadir = fullfile(dataDir, 'covtype.data');
A = importdata(datadir); % [581012, 55] the first column is index
labels = A(:, 55);
labels = labels'; % [1 581012]
data = A(:, 1:54);
data = data'; % [54 581012]

% set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
% data = single(reshape(cat(3, x1, x2),28,28,1,[]));  % catenate training data and test data together
% dataMean = mean(data(:,:,:,set == 1), 4);
% data = bsxfun(@minus, data, dataMean) ;

imdb.data = data;
imdb.labels = labels;
% imdb.images.data_mean = dataMean;
% imdb.images.labels = cat(2, y1, y2) ;
% imdb.images.set = set ;
% imdb.meta.sets = {'train', 'val', 'test'} ;
% imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
