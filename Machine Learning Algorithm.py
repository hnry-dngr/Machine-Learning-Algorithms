# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:07:51 2024

@author: HP
"""
# # #Part1: supervised learning

# Import you data and perform basic data exploration phase
# Display general information about the dataset
# Create a pandas profiling reports to gain insights into the dataset
# Handle Missing and corrupted values
# Remove duplicates, if they exist
# Handle outliers, if they exist
# Encode categorical features
# Prepare your dataset for the modelling phase
# Apply Decision tree, and plot its ROC curve
# Try to improve your model performance by changing the model hyperparameters


# # #Part2: unsupervised learning

# Drop out the target variable
# Apply K means clustering and plot the clusters
# Find the optimal K parameter
# Interpret the results

import warnings
import pandas as pd
import numpy as np
import sweetviz as sv
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


warnings.filterwarnings("ignore")

data = pd.read_csv("C:/Users/HP/Downloads/Microsoft_malware_dataset_min.csv")
data1 = pd.read_csv("C:/Users/HP/Downloads/Microsoft_malware_dataset_min.csv")

# Exploratory Data Analysis
data.info()
data_head = data.head()
data_tail = data.tail()
data_descriptive_statistic = data.describe()
data_more_desc_statistic = data.describe(include = "all")
data_mode = data.mode()
data_distinct_count = data.nunique()
data_correlation_matrix = data.corr()
data_null_count = data.isnull().sum()
data_total_null_count = data.isnull().sum().sum()
data_hist = data.hist(figsize = (15, 10), bins = 20)

# my_report = sv.analyze(data)
# my_report.show_html()


# Removing Missing values
data = data.dropna()
# imputer = SimpleImputer(strategy= "most_frequent")
# imputer.fit(data)
# new_data = imputer.fit_transform(data)
# new_data = pd.DataFrame(new_data, columns= imputer.feature_names_in_)


data = data.drop(["CountryIdentifier", "OsPlatformSubRelease", "Census_OSEdition"], axis = 1)
data_new = data

# x=data.drop(["HasDetections"], axis=1)
# y= data["HasDetections"]

# Split the dataset into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


# classifier = DecisionTreeClassifier()  
# model = classifier.fit(x_train, y_train)   #fitting our model
# y_pred_train = model.predict(x_train)  
# y_pred_test = model.predict(x_test)
# accuracy_test = accuracy_score(y_test, y_pred_test)

# classification_report_train = classification_report(y_train, y_pred_train)
# accuracy_train = accuracy_score(y_train, y_pred_train)
# recall_train = recall_score(y_train, y_pred_train)
# precision_train = precision_score(y_train, y_pred_train)
# f1score_train = f1_score(y_train, y_pred_train)


# classification_report_test= classification_report(y_test, y_pred_test)
# accuracy_test = accuracy_score(y_test, y_pred_test)
# recall_test = recall_score(y_test, y_pred_test)
# precision_test = precision_score(y_test, y_pred_test)
# f1score_test = f1_score(y_test, y_pred_test)



# # Calculate ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# # Calculate Area Under the ROC Curve (AUC)
# auc = roc_auc_score(y_test, y_pred_test)

# # Plot ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
# plt.plot([0, 1], [0, 1], color='red', linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Windows machine’s probability of getting infected by various families of malware (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()

# Model Training
store_inertia = []
store_model = {}
clusters = []
for num in range(1, 21):
    clusters.append(num)
    
    clusterer = KMeans(n_clusters = num, random_state = 0)
    model = clusterer.fit(data_new)
    
    store_model[f"Cluster_{num}"] = model # Optional
    store_inertia.append(model.inertia_)

plt.figure(figsize = (15, 10))
plt.plot(clusters, store_inertia, marker = "o", linestyle='dashed',)
plt.xlabel("Number of clusters")
plt.xticks(np.arange(0, 21, 1))
plt.ylabel("WCSS")
plt.title("Elbow Diagram")
plt.show()


# Model Prediction
y_pred = store_model["Cluster_5"].labels_

# Model Evaluation
# -----> ASSIGNMENT (Find out type range of values for the METRICS and what they mean.)
metric_silhouette = silhouette_score(data, y_pred)
metric_davies_bouldin = davies_bouldin_score(data, y_pred)
metric_calinski_harabasz = calinski_harabasz_score(data, y_pred)


# Graph of Clusters
# METHOD 1
select0 = data_new[y_pred == 0]
select1 = data_new[y_pred == 1]
select2 = data_new[y_pred == 2]
select3 = data_new[y_pred == 3]
select4 = data_new[y_pred == 4]
centriods = store_model["Cluster_5"].cluster_centers_

plt.figure(figsize = (15, 10))
plt.scatter(select0.iloc[:, 0], select0.iloc[:, 6], c = "red", s = 300, label = "High Income - Low Spenders")
plt.scatter(select1.iloc[:, 0], select1.iloc[:, 6], c = "blue", s = 300, label = "Sensible Spenders")
plt.scatter(select2.iloc[:, 0], select2.iloc[:, 6], c = "green", s = 300, label = "High Income - High Spenders")
plt.scatter(select3.iloc[:, 0], select3.iloc[:, 6], c = "yellow", s = 300, label = "Low Income - High Spenders")
plt.scatter(select4.iloc[:, 0], select4.iloc[:, 6], c = "brown", s = 300, label = "Low Income - Low Spenders")
# plt.scatter(centriods[0, 0], centriods[0, 1], c = "black", label = "Centriods")
# plt.scatter(centriods[1, 0], centriods[1, 1], c = "black", label = "Centriods")
# plt.scatter(centriods[2, 0], centriods[2, 1], c = "black", label = "Centriods", s = 20)
# plt.scatter(centriods[3, 0], centriods[3, 1], c = "black", label = "Centriods", s = 20)
# plt.scatter(centriods[4, 0], centriods[4, 1], c = "black", label = "Centriods")
plt.xticks(np.arange(-1, 3, 0.5))
plt.yticks(np.arange(-1, 3, 0.5))
plt.title("Analyzing different customer grouping in our business.")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
