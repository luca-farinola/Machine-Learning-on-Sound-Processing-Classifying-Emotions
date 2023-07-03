# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:40:05 2023
@author: lukfa
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np
import os

os.chdir('C:/Users/lukfa/OneDrive/Desktop/Project2/data')

# Import files
df_1 = pd.read_csv('X_CI_slow.csv')
df_2 = pd.read_csv('X_CI_opt_slow.csv')  # Load the third dataset
df_3 = pd.read_csv('X_slow.csv')
df_Y = pd.read_csv('y.csv')

df_Y['y_gender'] = np.where((df_Y['y_actors'] == 3) | (df_Y['y_actors'] == 4) | (df_Y['y_actors'] == 6), 'Male', 'Female')

# take first PCs 
#from sklearn.decomposition import PCA
#pca = PCA(n_components=13)
#df_3 = pca.fit_transform(df_3)

clf = RandomForestClassifier(max_depth=3, random_state=42)

num_runs = 100  # Number of runs
precision_scores_1 = []
precision_scores_2 = []
precision_scores_3 = []  # New list for the third dataset

for _ in range(num_runs):
    X_train, X_test, Y_train, Y_test = train_test_split(
        df_1,
        df_Y['y_actors'],
        stratify=df_Y['y_emotions'],
        test_size=0.2
    )

    clf.fit(X_train, Y_train)
    precision_1 = precision_score(Y_test, clf.predict(X_test), average=None, zero_division=1)
    precision_scores_1.append(precision_1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        df_2,
        df_Y['y_actors'],
        stratify=df_Y['y_emotions'],
        test_size=0.2
    )

    clf.fit(X_train, Y_train)
    precision_2 = precision_score(Y_test, clf.predict(X_test), average=None, zero_division=1)
    precision_scores_2.append(precision_2)

    X_train, X_test, Y_train, Y_test = train_test_split(
        df_3,
        df_Y['y_actors'],
        stratify=df_Y['y_emotions'],
        test_size=0.2
    )

    clf.fit(X_train, Y_train)
    precision_3 = precision_score(Y_test, clf.predict(X_test), average=None, zero_division=1)
    precision_scores_3.append(precision_3)

# Calculate average precision scores and standard deviations
avg_precision_1 = np.mean(precision_scores_1, axis=0)
std_dev_1 = np.std(precision_scores_1, axis=0)

avg_precision_2 = np.mean(precision_scores_2, axis=0)
std_dev_2 = np.std(precision_scores_2, axis=0)

avg_precision_3 = np.mean(precision_scores_3, axis=0)
std_dev_3 = np.std(precision_scores_3, axis=0)

# List of class/category labels
labels = ['actor 1', 'actor 2', 'actor 3', 'actor 4', 'actor 5','actor 6']
#labels = ['Female','Male']
#labels = ['anger','disgust','fear','happiness','sadness']

# Set the width of each bar
bar_width = 0.25

# Create index array for the bars
index = range(len(labels))


# Plot the average precision scores for dataset 1 with error bars for standard deviation
plt.figure(figsize=(10, 6), dpi=1200)  # Increase dpi to improve quality of the plot
plt.bar(index, avg_precision_1, width=bar_width, yerr=std_dev_1, label='CI', capsize=4, color='seagreen')

# Plot the average precision scores for dataset 2 with error bars for standard deviation
plt.bar([i + bar_width for i in index], avg_precision_2, width=bar_width, yerr=std_dev_2, label='Opt_CI', capsize=4, color='royalblue')

# Plot the average precision scores for dataset 3 with error bars for standard deviation
plt.bar([i + 2 * bar_width for i in index], avg_precision_3, width=bar_width, yerr=std_dev_3, label='normal hearing', capsize=4,color = 'silver')

# Customize the plot
plt.xlabel('Classes')
plt.ylabel('Precision')
plt.title('Average Precision Comparison for Emotions in different models')
plt.xticks([i + bar_width for i in index], labels)
plt.legend()
plt.grid(axis='y', alpha=0.5)


# Display the plot
plt.show()

