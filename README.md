# YouTube-Video-Recommendation-ML-Model
A Machine Learning group project for EECS 4404 that analyzes a dataset on the most trending videos using different ML techniques to determine what videos are generally going to be successful. Below is a report of our findings for this project. 


# Abstract
 
Our application is about a YouTube Video Recommendation system targeted specifically at content creators to determine the best approaches to creating successful videos. It uses various machine learning models and techniques to determine the best genres, metrics, and publication settings to use when creating videos to maximize performance of the video on YouTube (likes, views, watch-time, etc). The design revolves around 3 techniques: Decision trees, Clustering, and Regression to answer 3 different questions. With decision trees we like to consider the level of successfulness given video analytics, with clustering we want the best genre given video metadata, and with regression we would like to know which temporal features (time-based features) maximize viewership. We conclude that videos have a lot of correlated features that can affect how well it does on the platform; the larger the number of likes, dislikes and comments the more viral a video will become, gaming, entertainment, and music end up being some of most successful genres for video creation, and ideally publishing videos in the morning generally leads to better performance. These results aren’t fully definitive because of variations in the data and the limitations of our design/techniques but they provide a great basis for researching and developing better methodologies for video analysis.
# Introduction 
1. What is your application:
   
  Our application is a YouTube Video Recommendation system oriented toward video creators to allow them to get insight on their video ideas. The underlying machine learning models within our system would help video creators predict how successful a video will do based on metrics of similar videos and help them determine the best genres for video creation.

2. What are the assumptions/scope of your project:
   
  To make the analysis of data clearer and computation of models relatively quick, we focus our dataset to include videos only published in North America, specifically Canada and the USA. This is because we assumed Western videos would share more similarities and predictable patterns to analyze and extract due to the commonality in culture and the population compared to using international YouTube videos. Another scope to be aware is our definition of “successful”, in this project we use YouTube videos that appear on the daily Trending Video page on YouTube so the analysis done within our project help to find the best way a video will make it on the Trending Page and not just have some arbitrary number of views or likes. Finally, our dataset is limited in terms of recency as it contains top trending videos from 2017 to 2019 so the analysis and results obtained by our models cannot be guaranteed to reflect modern day video trends, however it is possible to extrapolate our predictions to see if they compare to today’s standards.

3. Adjustments to Original Project Plan:
   
  Our original topic and goal have stayed the same from Part 1, as well as the questions we were hoping to answer. However, our approaches and technical aspects of our underlying machine learning models for these applications has changed. Our decision tree approach to classify the category of successfulness has mostly stayed the same except that we changed it from a binary classification model to a multi-class classification model to better divide the data and highlight differences in videos. We changed our clustering model to use K-means rather than DBSCAN as it produced more clearer results and was easier to implement, but the target question to answer still stayed the same. Our regression model changed by excluding runtime as a feature because of the difficulty of scraping that data. More specifically, our dataset was so large that obtaining and preprocessing video runtime was not computationally efficient. Our PCA model also stayed the same. In short, the topic of our application and project has stayed the same but our implementations details have changed where the reason for which will be elaborated in the sections below. We ended up not using YouTube API calls. Reasons are the dataset provides adequate data features for the model training purpose, and YouTube sets a 10,000-point quota of its API calls, and some intricate data query for our dataset can easily use up the quota.

# Methodology
1. Design/pipeline
   
 Decision Tree:
1) Import our dataset and use the views, likes, dislikes, comment_count columns as features.
2) Preprocess data by manually labelling the data using thresholds into 1 of 4 classes (Unpopular, Average, Popular, Viral).
3) Encode the labels using a label encoder so labels are numeric for easier processing
4) Randomize and split the data into 80% for training and 20% for testing
5) Train the decision tree and make predications; measure accuracy using F-Score and refine model as needed (iterate back to step 2 Visualize the best decision tree model.

Clustering:

1) Import our dataset and use the titles, tags, descriptions columns as features.
2) Preprocess data by replacing null values with empty strings and vectorize textual data using TF-IDF. We use 1-3 words for key
phrases and use a column transformer to combine vectorized features
3) Experimentally try different number of clusters and graph the Silhouette Score and WCSS to determine the optimal k for K-means
4) Predict clusters using optimal k and visualize clusters using a WordCloud, depending on results modify step 2 or try different k
values.

Regression:

1) Import our dataset and extract the video_id, publish_time, likes, dislikes, comment_count, views columns
2) Preprocess data by web scrapping runtime of videos using video_id and converting publish_time into publish_hour (24-hour
format). Use these 2 new features plus the likes, dislikes, comment_count as features and views as labels.
3) Randomize and split the data into 80% for training and 20% for testing. Standardize features so L1 (Lasso) Regularization can be
used.
4) Perform Grid Search Cross Validation to determine best alpha hyperparameter; train model with this alpha and make
predications.
5) Use R2 score, Median and Mean Absolute Error to measure accuracy/evaluate model. Plot a Heat-Map and Pair Plots for feature
pairs to determine correlations and cross reference results with PCA output. Use the key information from both these evaluation
methods to iterate back to steps 2-4 to refine model.

PCA:

1) Import a dataset, preprocess by trim out invalid video entries (comments disabled, rating disabled, video error)
2) Augment the dataset with 2 more meaningful features (net likes, view-interactivity rate) deduced from existing features.
3) Prepare the integer data of each video entry by log10 to approximate normal distribution. And then shift each feature’s mean to
4) Perform PCA calculation and find the number of components that accumulates explained variance to 95%, and the corresponding
feature weight vectors.
5) Retrieve the dimensionally reduced new dataset.
6) Repeat above for the other dataset
   
2. Dataset
   
 We obtained our dataset from a public dataset on Kaggle called “Trending YouTube Video Statistics”. Here based on our assumptions we only used the “CAvideos.csv” and “USvideos.csv” files for our model’s dataset which equates to 81830 unique trending YouTube videos to potentially use for training. The public dataset comes with 16 different categories to possibly use as features or labels; in our models we use a subset of 8 of the categories in our dataset which are views, comment_count, likes, dislikes, tags, titles, descriptions, publish_time. For our decision tree model, we used views, comment_count, likes, dislikes as our features, where our pre-processing consisted only of shuffling the data and splitting 80% of it for training and 20% into testing. This is because these 4 features were in ranges of around the same order of magnitudes so feature scaling was not required. For our K-means clustering we used tags, titles, descriptions as features and our preprocessing included using a text vectorizer to turn the video metadata into numeric features for the model to do calculations, filling any null values with empty strings and running the K-means algorithm for k values between 2 to 10 to find the optimal number of clusters. This was done by graphing a WCSS vs Clusters plot and a Silhouette Score (SS) vs Clusters plot, in our case k=10 minimized the WCSS and maximized the SS. Finally, our regression model used publish_time, views, comment_count, likes, dislikes as features and our preprocessing included extracting only the hour from publish_time (24-hour format) to generalize the time of day, and we ran a Grid Search Cross Validation algorithm to find the best alpha/lambda value for our L1 (Lasso) regularization parameter since we wanted to perform feature selection within our linear regression model.
 

3. Model training
   
 To classify videos into different tiers of successfulness based on their analytics we used a Decision Tree model for the Multi-Class Classification problem:
- Inputs (features): views, comment_count, likes, dislikes
- Technique Details: The labels for our training data were created through thresholds or ranges applied to each feature,
for example if views >= 106 then this would mean a “Viral”. These thresholds were selected by looking at the mean and median of each of the features and trying to find commonalities between values of different features. Once we had the labels for each video, we used a label encoder to turn them into numeric labels with the following map: [“Viral” = 3, “Popular” = 2, “Average” = 1, “Unpopular” = 0]. Now in our training phase we used 80% of the dataset (65464 videos) and 20% (16366) for testing/predicting.
- Outputs: When making predictions the output of the model will be 0, 1, 2 or 3 representing the label/class.
  
To determine the best genres and types of videos that make a video trending we used K-Means Clustering to cluster videos based on keywords found within their metadata/textual data:
- Inputs (features): titles, description, tags
- Technique Details: We used text vectorization to convert key phrases of 1-3 words in length to numeric data (TF-IDF) to
allow for processing when we cluster. If metadata was missing from a video, we ensured it had a default value of blank or 0 so the clustering algorithm didn’t break. We manually ran K-Means for k values between 2 and 10 which were arbitrarily chosen and found 10 to be ideal and still computationally efficient. The true model was then trained with 10 clusters for 600 iterations with 10 centroid seed randomizations.
- Outputs: The 10 clusters where each video had a cluster label and a WordCloud for cluster visualization.
  
To determine the relationship between number of views and video metrics including analytics but also publish time we used Multiple Linear Regression to create a way to predict views given such metrics:

- Inputs (features): comment_count, likes, dislikes, publish_time and views (labels for training)
- Technique Details: publish time was converted only to hours in 24-hour format so it was easier to use as a feature and draw conclusions from (ex. what time of day do views get maximized). Before training our data, we had to standardize
our features so we could use L1 regularization and use Cross-Validation (CV) training to find the best alpha value which
in our case was 100.0. We had 5 folds for the CV and had an 80/20 split for the train/test sets respectively.
- Outputs: The predicated number of views.
  
To simplify model training or find trends within similar features we use PCA’s closed form solution of a preprocessed dataset that looks for the list of principal components and their explained variances.

- Inputs (features): views, likes, dislikes, comment_count, and deduced likes_net, view_interact_rate
- Technique Details: The model directly uses PCA’s closed form solution of a preprocessed dataset and looks for the list of
principal components and their explained variances. The raw number data are log_10 and mean 0-shifted to resemble
normal distributions.
- Outputs: The reduced new features, and the weighted vectors to calculate from the input features.
  
4. Prediction
   
 For our decision tree model, it is a binary tree, so it splits twice for a given internal node which makes sense based on our threshold rule (>= or <), each split is determined by the Gini Index where the lowest for a given feature is selected for a split. For our clustering model, once the 10 clusters are computed we can run predictions on the model for new videos it has never seen given their title and the model will output the cluster it belongs to (integer from 0-9). To predict the exact genre, the WordCloud can be used to map the cluster number to the genre. The regression model is trained for 1000 iterations where we used Coordinate Descent as the cost function, since it is computationally efficient and handles Lasso regularization better than gradient descent. Since the regression model also is in higher-dimensional space (many features), we cycle through each feature sequentially and update one of them in each iteration. Predictions are simply made with the multi-linear model equation obtained from training via substitution. Our PCA for the features is an unsupervised machine learning model so currently it won’t produce predictions on new dataset. However, the results show obvious similarity between the 2 datasets, in terms of PCA components’ explained variance accumulations, and the similarity between their corresponding PCA components (weighted vectors).

# Results
1. Evaluation
   
 Our models are evaluated through measuring accuracy via different metrics and plotting graphs to obtain the relationships between features and the outputs of the models. The predictions for the supervised techniques are made on 20% of our dataset (16366 videos) while unsupervised techniques operate on the entire dataset. We separated the data into the test set by first shuffling then allocating data into the test set.

1) Decision Tree:
- Based on our manually labelling for the training data, we ended up with skewed dataset that was concentrated
around unpopular videos, to compensate for this we used F-Scores, balanced accuracy, and Confusion matrices as
our metrics.
- Our model performed exceptionally well with F-Scores of [0.99991, 0.9996, 0.998, 1] for classes [0, 1, 2, 3]
respectively. The balanced accuracy (average of recalls) was 99.9% among all tests, and Confusion matrices
indicated only 1-2 samples were misclassified for classes 0, 1 and 2.
- This is expected as with the manually labelling and clear numeric features, human performance would be near
perfect as well. If time permitted, we would have liked to compare our results with a Random Forest implementation of the classifier. We also visualized the tree which showed us our trained tree had height 11 and 25 leaf nodes.

2) K-Means Clustering:
- Our baseline here was also human performance as humans can differentiate between topics/genres very well by
looking at the text and analytics.
- When finding the optimal number of clusters which was 10 for us were graphed charts of the Silhouette scores and
WCSS for clusters 2 to 10, we found that k=10 produced a maximum SS = 0.007 and minimum WCSS = 235500; when graphed in the range 2 to 20, we could also see that k=10 was the elbow point and too many clusters would over divide the data.
- Finally, we visualized the 10 clusters using a WordCloud which showed us the genres that each cluster corresponds to, for example cluster 1 = music, 3 = gaming, 6 = entertainment, 8 = news/politics, etc.

3) Multi-Linear Regression (Picture of Revised Polynomial Regression below as MLR had too many features to visualize): o Here we used many metrics to understand our model better, we mainly focused on the R2 score and Median
Absolute Error (MAE) where our R2 score = 0.786 and MAE = 299060. This means there are larger variances in the
data that are not accounted by the current features.
- To better understand the correlations between features we generated a Heat-Map and Pair-Plots for each pair of
features.
- Comparing the R2 score with the explained variances in our PCA we see that there is a discrepancy between the
explained variances of our model and the principal components of our data.
Here are some images of our Results, Graphs and Visualizations. For all the images we generated you can look inside our zipped project file or our GitHub Link to find them.
  

2. Results

Our decision tree and K-means clustering performance exceptionally well on the testing and all data and was aligned with our expectations. However, our regression model performed decently on the test set but was not aligned with our expectations. PCA indicates that views, likes and dislikes are the highest weighted features that affect the explained variance in our 3 principal components, but at least 20% of those changes aren’t reflected in the regression model. Our MAE value was larger than expected and based on our coefficients of [- 121976, 5890012, 1576244, -2300928] for features publish_hour, likes, dislikes, and comment_count respectively, and our Heat-Map there are a lot of negative correlation between features. The PCA results show that at most 3 combinations of the target 6 video features can successfully explain more than 95% of the dataset variance and numerical features. If the explanation threshold is 80%, just 1 combination of the given features is enough to represent the numerical features of a trending video, or we can name it “video trending index”.

# Discussions
1. Implications

 Our decision tree provided very well results and we attribute them to finding good thresholds for labelling the data and specifically sticking with numeric features, having textual data may have required a more complex tree structure (sensitive to outliers/over-fitting) which in our design was unwanted. Our clustering model also ended up performing better than we hoped as we assumed due to the larger variation in types of videos in our dataset, the clustering algorithm may have issues pinpointing easily differentiable clusters. However, it turned out to perform well due to our preprocessing to find the optimal k and vectorization of data. Our regression did not perform well and there could many reasons for why. One is that the feature scaling should have been done to scale down likes, dislikes, and comment_count to the order of magnitudes of publish_time since its range of [0, 24] is small. Another could be the inverse correlation between video analytics and temporal features so it may have been better to perform 2 regression models where one used only runtime and publish_time as features and the other used likes, dislikes, and comment_count. This idea is also reflected in our PCA where temporal features contributed less overall variance to the data since their weights were lower for the 3 principal components. As for PCA data processing, since all the input values are results of human interactivities with the trending videos from various channels, the raw number data may be heavily skewed because of those viral videos’ statistics. The successful dimension reduction of dataset features indicates that the log10 preprocess of the viewer related numerical features of a YouTube video is a good approach in general. It transforms data of Poisson distribution as Normal distribution-like, which is suitable for PCA processing and other regression techniques. The trending videos from Canada and the U.S. shares similar video characteristics, that their respective major principal components (the 1st principal component vector that explains about 80%) are very similar.
6

2. Strengths
 
 Some strengths of our design include using different plots to visualize data and relationships, using structured numeric data for our decision tree, vectorizing video metadata for training and using multiple metrics when assessing our models. Our most important strength during this project was being able find weaknesses in our models and use other techniques to offset them. For example, for our regression model once we saw the issues the multi-linear regression model, we decided to perform polynomial regression to compensate for some of the non-linear relationships between the features we used. We performed feature scaling on views via normalization, only used publish_time as a feature, used a dev set to iterate over multiple polynomial models and removed any outlier data to fix the issues mentioned before. Overall, this led to consistent and clearer conclusions where we obtained 7.4 (7 am) as the best time to publish a video to maximize viewership. PCA result shows 50% reduction on the number of numerical features to describe a video and it potentially suggests a video trending index to help content creators understand the numbers behind popular videos (also makes feature selection clearer and more apparent for the us developing the models). Another strength of our design is that our models can be easily extrapolated to handle similar problems for other social media platforms. Due to the similarities in the features between social media platforms our models can be easily modified to handle other content creation analysis for videos/posts on other platforms like Instagram.
 
3. Limitations
 
 Some limitations of our design include poor optimization for our multi-linear regression model, possibly some overfitting within our decision tree, and duplicate clusters produced from K-means. With the regression as mentioned before we wanted to use runtime as a feature but scraping the runtime using the YouTube API ended up being too computationally extensive to run alongside the regression model. This could also prevent us from reaching better conclusions about the relationship between views and temporal features since a key feature was missing in the analysis. Our decision tree even though it performed well, seemed to score too high especially with the skewed dataset. This could imply some overfitting which we did not address due to time constraints. The current PCA model didn’t consider video runtime as another numerical video feature, it may not have needed log10 preprocessing. With clustering even though clear genres were discovered, a lot of the clusters had similarities between them we wanted to divide as multiple videos could be ambiguously labelled to multiple clusters which meant the model was not accurate enough. Finally, if this project were to be extended for present YouTube, then dislikes would not be a valid feature since they are not publicly available to extract, one solution to this may be performing sentiment analysis on the comments of video watchers.
 
4. Future directions
 
 If we continue to work on this project in the future, some things we would like to consider are trying a Random Forest model for the multi-class classification to see if they are some overfitting issues with the training. We also like to attempt DBSCAN and compare the results to K-means as it is better equipped with ignoring outliers and aggregating dense data points (which we have a lot of) into independent clusters. We would also like to continue our polynomial regression and build off its better performance compared to our original regression. Specifically, implement hyperparameter tuning via Grid Search Cross Validation and use L1 regularization to address overfitting/feature selection (especially when we include runtime and numeric features). Another consideration is to attempt to use NNs as our model to answer our 3 questions because of the large amounts and non-linear relationships within our data. We didn’t do this initially as we did not expect a large degree of variations within the data and assumed the computation resources/time required for training would be too much. Finally, to make the application really viable, we would like to expand the video datasets beyond only North American videos and incorporate international videos as well. Some changes to our current model we would also like to try is using sentiment analysis to extract feelings from user comments since in present day YouTube databases don’t publicly show the dislike count so this feature could replace the dislike feature. Another idea is to find a way to make our models continuous learn throughout time; YouTube trending data is always changing so finding a way to have our models adapt overtime by training with new examples after fixed intervals of time could help our model stay relevant and up to date for accurate results.

# References
1. Raw Dataset from Kaggle: https://www.kaggle.com/datasets/datasnaek/youtube- new/data?select=CAvideos.csv
2. Approaches to Data Splitting: https://snji-khjuria.medium.com/everything-you-need-to-know- about-train-dev-test-split-what-how-and-why-6ca17ea6f35
3. Decision Tree Template Code: https://www.datacamp.com/tutorial/decision-tree-classification- python
4. Ways to modify Decision Trees/Random Forest: https://developers.google.com/machine- learning/decision-forests/random-forests
5. Implementing K-Means: https://www.youtube.com/watch?v=N0o-Bjiwt0M
6. Lasso vs Ridge Regularization: https://www.datacamp.com/tutorial/tutorial-lasso-ridge-
regression
7. PyTube Documentation: https://pytube.io/en/latest/api.html#youtube-object
8. Sklearn Lasso Documentation: https://scikit-
learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Las
so
9. Sklearn Grid Search Documentation: https://scikit-
learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
10. Error Metric Suggestions: https://stats.stackexchange.com/questions/131267/how-to-interpret-
error-measures
11. YouTube Video Average Runtime Baseline:
https://www.statista.com/statistics/1026923/youtube-video-category-average- length/#:~:text=According%20to%20the%20report%2C%20the,of%206.8%20minutes%20per%2 0video.
12. Linear Regression Template Code: https://scikit- learn.org/stable/auto_examples/linear_model/plot_ols.html
13. Regularization Suggestions: https://medium.com/@zxr.nju/the-classical-linear-regression- model-is-good-why-do-we-need-regularization- c89dba10c8eb#:~:text=The%20three%20most%20popular%20ones,%2C%20Lasso%2C%20and% 20Elastic%20Net.&text=Ridge%20regression%20is%20also%20called,function%20of%20the%20 squared%20coefficients.
14. Training via Coordinate Descent: https://stats.stackexchange.com/questions/146317/when- should-one-use-coordinate-descent-vs-gradient- descent#:~:text=Coordinate%20descent%20updates%20one%20parameter,do%20better%20tha n%20the%20other.
15. Pandas Documentation: https://pandas.pydata.org/docs/
16. MatplotLib Documentation (Consulted for Visualizations):
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
17. Seaborn Documentation (Consulted for Visualizations): https://seaborn.pydata.org/
18. Sklearn Documentation (Consulted for many things): https://scikit-learn.org/stable/
19. Sklearn PCA model: https://scikit-
learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.P CA
