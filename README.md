# Recommender_Systems
Exploration of recommender system algorithms and their various implementations and applications

This serves as a general repository for my explorations of various Recommender System algorithms in MATLAB, Python, and R. 

General overview: 
Recommender systems are algorithmic approaches to providing personalized recommendations to users based on past behaviors, preferences, and interests. There are three common types of RS: 
1) Content-based filtering (CBF): This approach recommends items/services based on the user's history of liking or interaction. 

2) Collaborative filtering (CoFi): This approach recommends items/services based on the behavior or ratings of other users with similar preferences to the user. In this repository, examples of using CoFi in: 

      MovieRatings_CoFi (MATLAB folder) : This uses a vectorized implementation of a simultaneously updated gradient descent with respect to users and features (using              the `cofiCostFunction.mat` function) to recommend movies from a 1682-item list based on the user's rating of a small subset of movies and ratings from 943                other users 
      
3) Knowledge based systems (KB): This approach recommends items to users based on specifically sought attributes. Recommendations are based on similarities between user requirements and item attributes.    
      
      

