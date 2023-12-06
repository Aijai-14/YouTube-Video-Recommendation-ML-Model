import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

# Array of features
col_names = ['title', 'tags', 'description']

# Array of csv files for data
csv_files = ['CAvideos.csv', 'USvideos.csv']

# load and combine dataset from CAvideos.csv and USvideos.csv files
video_metadata = pd.concat([pd.read_csv(data) for data in csv_files])

# create dataframe using relevant features
video_metadata = video_metadata[col_names]

# this loop checks for null values in any of the feature columns and fill them with empty string
for column in video_metadata.columns:
    video_metadata[column].fillna("", inplace=True)

# create 3 text vectorizer objects that take turn the text data for each feature into numeric data
# for the clustering algorithm to work. We learned this from the first example video provided in Project Part 1.
textVectorizerTitles = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), strip_accents='unicode')
textVectorizerTags = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), strip_accents='unicode')
textVectorizerDescriptions = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), strip_accents='unicode')

# combine all 3 text vectorizers into 1 column transformer so that the vectorizer can be applied on the entire
# dataframe. We added this code to make the vectorized text into 1 object, so we could add it to dataframe.
column_transformer = ColumnTransformer(
    [('tf_title', textVectorizerTitles, 'title'),
     ('tf_tags', textVectorizerTags, 'tags'),
     ('tf_description', textVectorizerDescriptions, 'description')], verbose_feature_names_out=False)

# turn the text features into numeric features
features_trans = column_transformer.fit_transform(video_metadata)


# Loop through the KMeans cluster algo. and using the specified number of clusters from 2 to 10 and store the
# sum of squared distances for each number of cluster and the silhouette scores for each cluster. We build this function
# off the function show in the first example video provided in Project Part 1, but added our dataset and code to plot/calculate
# the SS and WCSS.

def findOptimalCluster():
    maxClusters = 10
    number_of_clusters = range(2, maxClusters + 1)
    KmeansInertia = []
    silhouette_scores = []
    for cluster in number_of_clusters:
        K = KMeans(n_clusters=cluster, n_init=10)
        K.fit(features_trans)
        KmeansInertia.append(K.inertia_)
        silhouette_scores.append(silhouette_score(features_trans, K.labels_, metric='euclidean'))

    # plot the sum of squared distances vs number of cluster graph.
    plt.plot(number_of_clusters, KmeansInertia, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Clusters vs Distance')
    plt.savefig('Squared Distances vs Clusters Graph')

    # plot the silhouette score vs number of cluster graph.
    plt.plot(number_of_clusters, silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Clusters')
    plt.savefig('SS vs Clusters Graph')

# The below code was built off of the first example video provided in Project Part 1 but with our own dataset and features and
# model parameters.
# Based on the graphs created we fit the data on the KMeans model with the ideal number of clusters
optimalCluster = 10
trueModel = KMeans(n_clusters=optimalCluster, init='k-means++', max_iter=600, n_init=10)
trueModel.fit(features_trans)

# After clustering the data we can obtain the predicated labels/clusters each video belongs to and display them in a
# dataframe.
predictedClusters = trueModel.labels_
videoClusters = pd.DataFrame(list(zip(video_metadata['title'], predictedClusters)), columns=['title', 'cluster'])
print(videoClusters.sort_values(by=['cluster']))

# We can use the WordCloud library to visualize which video titles belong to which cluster and get an idea of the
# genre the cluster represents.
for k in range(optimalCluster):
    videoTitles = videoClusters[videoClusters.cluster == k]['title'].str.cat(sep=' ')
    wordCloud = WordCloud(max_font_size=50, max_words=100, background_color='white').generate(videoTitles)

    plt.subplot(2, 5, k+1).set_title('Cluster ' + str(k))
    plt.plot()
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')

plt.savefig('Cluster Visualization')
