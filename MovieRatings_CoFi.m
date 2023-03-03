% This implementation is an example of a Memory-Based Method of collaborative filtering employing User-Based and Item-Based explicit ratings data

% This script will go through how to use relatively sparse user ratings and movie features
% data to recommend movies that a new user might like. This implementation
% uses a simple vectorized gradient descent algorithm to fit the model. 

% The framework for this example as well as the data set were used in the
% Machine Learning Coursera Course offered by Stanford University
% (Instructor Andrew Ng)

%% ============== Load the movie ratings and dataset=======================
fprintf('Loading movie ratings dataset. \n')
load('movie_ratings.mat');

% The dataset consists of 1683 movies and 943 users. 
% R is a logical matrix of movies x users, where 1 indicates that the movie
%       was rated by the user (1 = yes, 0 = no)
% Y is a matrix of movie ratings on a 1-5 scale, zero indicates that the
%       movie was not rated by the user (redundant with R matrix)

imagesc(Y)
ylabel('Movies')
xlabel('Users')

%% ==================== Get new user ratings for movies ===================
% To train the Collaborative filtering model, we need to get ratings that
% correspond to the new user. To do so, we import the movie titles first:

fid = fopen('movie_ids.txt');
%Store all of the movies in a cell array
n = size(Y,1); %Total number of movies
movieList = cell(n,1);
for i = 1:n
    line = fgets(fid);
    [idx] = strtok(line, '');
    movieList{i} = strtrim(idx);
end
fclose(fid);

% Initialize new user ratings
my_ratings = zeros(n,1);
n_movies_to_rate = 10;
sample_movies = randi(1000,[n_movies_to_rate,1]);
% Collect new user ratings
fprintf('How do you rate the folling movies \n ([1-5]if you have seen it,0 if you have not)? \n\n')
for i = 1:length(sample_movies)
    my_ratings(sample_movies(i)) = input(movieList{sample_movies(i)});
end

fprintf('\n New user ratings: \n')
for i = 1:length(my_ratings)
    if my_ratings(i) > 0
        fprintf('Rated %d for %s \n', my_ratings(i), movieList{i});
    end
end

%% ======================= Learning movie ratings =========================
% Now to train the collaborative filtering model on the movie rating
% dataset provided

fprintf('Training the collaborative filtering model')
%adding the new user ratings to the model
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

% Normalize movie ratings
Ymean = zeros(n,1);
Ynorm = zeros(size(Y));
for i = 1:n
    idx = find(R(i,:) == 1);
    Ymean(i) = mean(Y(i,idx));
    Ynorm(i, idx) = Y(i, idx) - Ymean(i);
end

% Useful values
num_users = size(Y,2);
num_movies = size(Y,1);
num_features = 10; %You can set this value however you like, start with 10

% Set initial parameters (Theta, x)
x = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [x(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set regularization 
lambda = 2; 
theta = fmincg(@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
    num_features, lambda)), ...
    initial_parameters, options);

% Unfold the returned theta back into U and W
x = reshape(theta(1:num_movies*num_features), num_movies, num_features); 
Theta = reshape(theta(num_movies*num_features+1:end), ...
    num_users, num_features);

fprintf('Recommender system learning completed. \n')


%% ================ Recommendations for New User ==========================
% computing the prediction matrix for new user input give recommendations
% with the trained model

p = x*Theta';
my_predictions = p(:,1) + Ymean;

fid = fopen('movie_ids.txt');
%Store all of the movies in a cell array
n = size(Y,1); %Total number of movies
movieList = cell(n,1);
for i = 1:n
    line = fgets(fid);
    [idx] = strtok(line, '');
    movieList{i} = strtrim(idx);
end
fclose(fid);

[r,ix] = sort(my_predictions, 'descend');
fprintf('Top Recommendations for New User: \n')
for i = 1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j),...
        movieList{j});
end

fprintf('\n Original ratings provided: \n')
for i = 1:length(my_ratings)
    if my_ratings(i) > 0
        fprintf('Rated %d for %s \n', my_ratings(i), movieList{i});
    end
end



