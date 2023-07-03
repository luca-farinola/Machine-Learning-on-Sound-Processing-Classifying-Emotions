# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:21:45 2023

@author: lukfa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
import os 

os.chdir('C:/Users/lukfa/OneDrive/Desktop/Project2/Data')

# Create an empty DataFrame to store feature selection results
feature_selection_df = pd.DataFrame(columns=['Iteration', 'Selected_Features'])

# Define the number of iterations to run the code
num_iterations = 50

for i in range(num_iterations):
    df_X = pd.read_csv('X_slow.csv')
    df_Y = pd.read_csv('y.csv')

    # Split the dataset into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        df_X,
        df_Y['y_emotions'],
        stratify=df_Y['y_actors'],
        test_size=0.2,
    )

    clf = RandomForestClassifier(max_depth=3, random_state=0)
    clf.fit(X_train, Y_train)

    # Get the feature importances
    feature_importance = clf.feature_importances_

    # Sort feature importance in descending order
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_importance = feature_importance[sorted_indices]
    sorted_features = X_train.columns[sorted_indices]

    # Set the threshold for feature importance
    threshold = 0.005

    # Filter the feature importances and corresponding feature names based on the threshold
    filtered_importance = sorted_importance[sorted_importance > threshold]
    filtered_features = sorted_features[sorted_importance > threshold]

    # Store the selected features in the DataFrame
    iteration_df = pd.DataFrame({
        'Iteration': i + 1,
        'Selected_Features': [filtered_features.tolist()]
    })
    feature_selection_df = pd.concat([feature_selection_df, iteration_df])

# Count the occurrence of each feature and calculate its total importance across iterations
feature_counts = feature_selection_df['Selected_Features'].explode().value_counts()
feature_importance_totals = feature_selection_df['Selected_Features'].explode().groupby(
    feature_selection_df['Selected_Features'].explode()).size()

# Filter the features based on their occurrence and total importance
selected_features = feature_counts[feature_counts >= (num_iterations / 2)].index
selected_features_importance = feature_importance_totals[selected_features]


# Create a DataFrame to display the selected features, their total importance, and scores
selected_features_df = pd.DataFrame({
    'Total_Importance': selected_features_importance,
})

print(selected_features_df)
