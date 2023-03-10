function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% Initialize outputs
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% Compute the cost function and gradient for collaborative
%               filtering. 
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
h = (X*Theta').*R;
yr = Y.*R;
 
J = sum(sum((h-yr).^2))/2 +...
    ((lambda/2)*(sum(sum(Theta.^2)))) +...
    ((lambda/2)*(sum(sum(X.^2))));


for j = 1:size(Theta,2)
    X_grad(:,j) = (h-yr)*Theta(:,j) + lambda*X(:,j);
end
for i = 1:size(X,2)
    Theta_grad(:,i) = (h-yr)'*X(:,i) + lambda*Theta(:,i);
end

grad = [X_grad(:); Theta_grad(:)];

end
