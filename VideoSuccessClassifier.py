import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support  # roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree


# function used to create labels for our dataset
def createLabels(training_data):
    if (
            training_data['views'] >= 1000000 and
            (training_data['likes'] >= 50000 or
             training_data['dislikes'] >= 2000) and
            training_data['comment_count'] >= 5000
    ):
        return "Viral"
    elif (
            500000 <= training_data['views'] < 1000000 and
            (25000 <= training_data['likes'] < 50000 or
             1000 <= training_data['dislikes'] < 2000) and
            2500 <= training_data['comment_count'] < 5000
    ):
        return "Popular"
    elif (
            150000 <= training_data['views'] < 500000 and
            (5000 <= training_data['likes'] < 25000 or
             250 <= training_data['dislikes'] < 1000) and
            1000 <= training_data['comment_count'] < 2500
    ):
        return "Average"
    else:
        return "Unpopular"


# custom mapping for labels for dataset
labelMap = {
    "Viral": 3,
    "Popular": 2,
    "Average": 1,
    "Unpopular": 0}

# Array of features
col_names = ['views', 'likes', 'dislikes', 'comment_count']
# Array of csv files for data
csv_files = ['CAvideos.csv', 'USvideos.csv']

# load and combine dataset from CAvideos.csv and USvideos.csv files
video_analytics = pd.concat([pd.read_csv(data) for data in csv_files])

# create dataframe using relevant features
video_analytics = video_analytics[col_names]

# Manually label data using createLabels function and encode the string labels into numeric labels using a custom label
# mapping
video_analytics['labels'] = video_analytics.apply(createLabels, axis=1)
labelEncoder = LabelEncoder()
labelEncoder.classes_ = video_analytics['labels'].unique()
video_analytics['Encoded_Labels'] = video_analytics['labels'].map(labelMap)

# split data into inputs and target outputs
features = video_analytics[col_names]
target_labels = video_analytics['Encoded_Labels']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, target_labels, test_size=0.2, random_state=21,
                                                    shuffle=True)

# Create Decision Tree object
classifier_tree = DecisionTreeClassifier()

# Train Decision Tree
trained_tree = classifier_tree.fit(X_train, y_train)

# Predict the classes for test dataset
predictions = trained_tree.predict(X_test)

# calculates confusion matrix for each class
confusion_matrix = multilabel_confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3])

# Calculate performance of model
accuracy = metrics.accuracy_score(y_test, predictions)
balanced_accuracy = metrics.balanced_accuracy_score(y_test, predictions)
precision, recall, F_Score, support = precision_recall_fscore_support(y_test, predictions, labels=[0, 1, 2, 3])
# AUROC = roc_auc_score(y_test, predictions, multi_class='ovr')

# display the performance metrics
print("Accuracy: ", accuracy)
print("Balanced Accuracy: ", balanced_accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F_Score: ", F_Score)
print("Support: ", support)
# print("AUROC: ", AUROC)
print("Confusion Matrices:")

label = 0
for matrix in confusion_matrix:

    if label == 0:
        print("Unpopular:")
    elif label == 1:
        print("Average:")
    elif label == 2:
        print("Popular:")
    else:
        print("Viral:")

    for row in matrix:
        print(row)
    print("")
    label += 1

# Visualize decision tree
plt.figure(figsize=(25, 20))
tree.plot_tree(trained_tree, fontsize=6)
plt.savefig('DecisionTree', dpi=100)
