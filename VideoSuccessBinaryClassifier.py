import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names = ['views', 'likes', 'dislikes']

# load dataset
video_analytics = pd.read_csv("USvideos.csv")

video_analytics = video_analytics[col_names]

print(video_analytics)

