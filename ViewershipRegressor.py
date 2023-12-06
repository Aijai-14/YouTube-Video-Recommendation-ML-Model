import pandas as pd
from pytube import YouTube
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

# Array of features
col_names = ['video_id', 'publish_time', 'likes', 'dislikes', 'comment_count', 'views']

# Array of csv files for data
csv_files = ['CAvideos.csv', 'USvideos.csv']

# load and combine dataset from CAvideos.csv and USvideos.csv files
videoData = pd.concat([pd.read_csv(data) for data in csv_files])

# create dataframe using relevant features
videoData = videoData[col_names]


# function to generate the runtime and publish hour feature. The runtime is obtained by using pyTube library to
# web scrape the duration using the video id and add it to an array. The publish hour is extracted by using publish time
# and parsing the hour component out then formatting in 24 hour format. Finally, both features are added to the videoData
# dataframe. We created this function from scratch.
def createFeatures():
    videoRuntimes = []
    for num in range(videoData.shape[0]):
        link = 'https://www.youtube.com/watch?v=' + videoData['video_id'].iloc[num]
        video = YouTube(link)
        videoRuntimes.append(video.length)

    publishHour = []
    for num in range(videoData.shape[0]):
        timestamp = (videoData['publish_time'].iloc[num])[10:]
        parsed_time = datetime.strptime(timestamp, "T%H:%M:%S.%fZ")
        formatted_hour = int(parsed_time.strftime("%H"))
        publishHour.append(formatted_hour)

    videoData['runtimes'] = videoRuntimes
    videoData['publish_hour'] = publishHour

    # any null values for runtime are filled with 702 seconds as a default value since that is the average duration of a
    # YouTube video in 2018.
    videoData['runtimes'] = videoData['runtimes'].fillna(702).replace([None, 'null', 'NaN'], 702)


# this function is similar to above but only computes the publish hour feature. We created this function from scratch.
def createTimeFeature():
    publishHour = []
    for num in range(videoData.shape[0]):
        timestamp = (videoData['publish_time'].iloc[num])[10:]
        parsed_time = datetime.strptime(timestamp, "T%H:%M:%S.%fZ")
        formatted_hour = int(parsed_time.strftime("%H"))
        publishHour.append(formatted_hour)

    videoData['publish_hour'] = publishHour

# create the features
createFeatures()

# split data into inputs and target outputs
features = videoData[['runtimes', 'publish_hour', 'likes', 'dislikes', 'comment_count']]
target_labels = videoData['views']

# The below code was built off of existing linear regression template from sklearn documentation but using our own dataset, features
# and model parameters.
# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(features, target_labels, test_size=0.2, random_state=1, shuffle=True)

# Standardize the features for Lasso regularization (sklearn's regularization implementation requires values to be centred
# around 0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# This code was researched and written by us to determine the best value of L1 regularization parameter alpha.
# Set up a range of alpha values for Grid Search
alphas = np.logspace(-4, 2, 10)
# Set up the parameter grid for GridSearchCV
param_grid = {'alpha': alphas}
# Create a Lasso model
lasso = Lasso()
# Perform GridSearchCV
grid_search = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_scaled, y_train)
# Get the best alpha from the grid search
best_alpha = grid_search.best_params_['alpha']
print(f"best_alpha = {best_alpha}")

# The below code was built off of existing linear regression template from sklearn documentation but using our own dataset, features
# and model parameters.
# create a LR model with lasso regularization parameter best alpha
lasso_model = Lasso(alpha=best_alpha)

# Fit the model to the training data
trained_model = lasso_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = trained_model.predict(X_test_scaled)

# Evaluate the model using various metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
Mae = median_absolute_error(y_test, predictions)
explained_var = explained_variance_score(y_test, predictions)

# The code below we wrote to print the metrics, coefficients, a pairplot and heat-map for our regression model. The sns
# library documentation was consulted for the visualizations.
# Print the metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Median Absolute Error: {Mae}")
print(f"Explained Variance Score: {explained_var}")

# Print the coefficients (feature weights) of the model
coefficients = trained_model.coef_
print("Coefficients:", coefficients)

# print the name of non-zero features (to see the effect of feature selection)
non_zero_features = [feature for feature, coef in zip(features.columns, coefficients) if coef != 0]
print("Non-zero features:", non_zero_features)

# graph pair plots for each pair of features
data_pairplot = pd.concat([X_train, pd.Series(y_train, name='target')], axis=1)
sns.pairplot(data_pairplot)
plt.savefig("PairPlot2")
plt.show()

# graph a heat map showing visualizing the correlation between each pair of features
correlation_matrix = X_train.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.savefig("Heatmap2")
plt.show()

# Without Runtime:
# Mean Squared Error (MSE): 8551762130950.944
# R-squared: 0.7862334648800452
# Mean Absolute Error (MAE): 931269.4602357434
# Root Mean Squared Error (RMSE): 2924339.605953957
# Median Absolute Error: 299060.07835400803
# Explained Variance Score: 0.786233944837189
# Coefficients: [ -121976.63875916  5890012.32634811  1576244.63085988 -2300928.08870579]
# Non-zero features: ['publish_hour', 'likes', 'dislikes', 'comment_count']

# ---------------------------------------------------------------------------

# With Runtime:
# best_alpha = 100.0
# Mean Squared Error (MSE): 8551761344160.955
# R-squared: 0.7862334845472632
# Mean Absolute Error (MAE): 931269.7740291776
# Root Mean Squared Error (RMSE): 2924339.4714295664
# Median Absolute Error: 299058.4221716494
# Explained Variance Score: 0.7862339645334453
# Coefficients: [ 7.32622660e+01 -1.21976206e+05  5.89001452e+06  1.57624477e+06
#  -2.30092924e+06]
# Non-zero features: ['runtimes', 'publish_hour', 'likes', 'dislikes', 'comment_count']
